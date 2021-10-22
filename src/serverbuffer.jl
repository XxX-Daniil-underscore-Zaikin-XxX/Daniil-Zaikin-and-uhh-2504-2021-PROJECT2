using Random, StatsBase, Parameters, LinearAlgebra, Distributions
import Base: isempty
include("simulate.jl")

Random.seed!(0)

struct Job
    unique_id::Int64
end

# function Job(unique_id::Int64)
#     return Job(unique_id)
# end

struct Server
    unique_id::Int
    buffer::Queue{Job}           # First element in buffer is currently served job
    max_buffer_size::Int         # Buffer size before overflow
    overflow_vector::Vector{Float64} # Probabilities of where an overflown job will go
    overflow_leave::Float64
    move_vector::Vector{Float64}     # Probabilities of where a finished job will go
    move_leave::Float64
    service_rate::Float64        # Mean service rate of this server
end

function Server(unique_id::Int, max_buffer_size::Int,       
    overflow_vector::Vector{Float64}, 
    move_vector::Vector{Float64},  
    service_rate::Float64)
    return Server(unique_id::Int, Queue{Job}(), max_buffer_size, overflow_vector, 1-sum(overflow_vector), move_vector, 1-sum(move_vector), service_rate)
end

struct ArrivalEvent <: Event
    job::Job
end

"""
    ArrivalEvent(unique_id::Int64)

Creates a new `ArrivalEvent` and populates it with a new `Job`; the `Job`'s `unique_id` is as given
"""
function ArrivalEvent(unique_id::Int64)
    return ArrivalEvent(Job(unique_id))
end

abstract type TransferEvent <: Event end

struct OverflowEvent <: TransferEvent
    transfer_to::Server
    job::Job
end

struct MoveEvent <: TransferEvent
    transfer_to::Server
    job::Job
end

struct LeaveEvent <: Event
    job::Job
end

# This event is called when the server finishes serving
struct ServedEvent <: Event
    server::Server
end

#The @with_kw macro comes from the Parameters package
@with_kw struct NetworkParameters
    L::Int
    gamma_scv::Float64 #This is constant for all scenarios at 3.0
    λ::Float64 #This is undefined for the scenarios since it is varied
    η::Float64 #This is assumed constant for all scenarios at 4.0
    μ_vector::Vector{Float64} #service rates
    P::Matrix{Float64} #routing matrix
    Q::Matrix{Float64} #overflow matrix
    p_e::Vector{Float64} #external arrival distribution
    K::Vector{Int} #-1 means infinity 
end


mutable struct DiscreteState <: State
    servers::Vector{Server}     # Enumerated list of servers
    left_jobs::Vector{Job}      # Enumerated list of jobs that have left the system
    moving_jobs::Vector{Job}    # Enumerated list of jobs that are moving across servers
    moving_jobs_num::Int        # Number of moving jobs (cheap memory-wise to keep them here)
    last_job::Int64             # Index of last-added job
    jobs_in_system::Int64       # Number of total jobs in sys (also cheap memory-wise)
    is_debug::Bool              
    params::NetworkParameters   # Hold our params in the state (is this a good idea? we'll find out)
end

"""
    DiscreteState(params::NetworkParameters)

Creates a DiscreteState, populates it with `servers` and `params` based on the given. First `job` index is 1, and all `job` storages start empty
"""
function DiscreteState(params::NetworkParameters)
    servers = [Server(i, params.K[i], Vector{Float64}(vec(params.Q[i:params.L:end])), Vector{Float64}(vec(params.P[i:params.L:end])), params.μ_vector[i]) for i in 1:params.L]
    return DiscreteState(servers, Vector{Job}(), Vector{Job}(), 0, 1, 0, false, params)
end

"""
Pull out the important info from our state - we don't need `params`
"""
function copy_state_items(state::DiscreteState)
    return deepcopy(state.servers), deepcopy(state.left_jobs)
end

"""
Returns whether a `Server`'s buffer is full
"""
function isfull(server::Server)
    return server.max_buffer_size != -1 && length(server.buffer) > server.max_buffer_size
end

"""
Returns whether a `Server` has no current `Job` and empty buffer
"""
function isempty(server::Server)
    return isempty(server.buffer)
end

"""
Returns index of `Server` a `Job` should transfer to, or 0 if the `Job` should leave
"""
function sample_probabilities(transfer_probs::Vector{Float64}, chance_leave::Float64)::Int
    return rand() < chance_leave ? 0 : sample(1:length(transfer_probs), Weights(transfer_probs))
end

"""
Returns server index a job should overflow to, or 0 if job should leave
"""
function get_overflow_index_from(server::Server)::Int
    return sample_probabilities(server.overflow_vector, server.overflow_leave)
end

"""
Returns server index a job should move to, or 0 if job should leave
"""
function get_move_index_from(server::Server)::Int
    return sample_probabilities(server.move_vector, server.move_leave)
end

"""
Returns the Server to transfer (overflow, move) Job to, or nothing if Job is to leave
"""
function get_transfer_from_index(state::DiscreteState, index::Int)
    return state.servers[index]
end

"""
Process a job leaving the system. Simply adds it to the state's array of left jobs
"""
function leave_job(state::DiscreteState, job::Job)
    state.jobs_in_system -= 1
    # push!(state.left_jobs, job)
    # # state.is_debug && println("Job #", job.unique_id, " left")
end

"""
Handles moving a Job between Servers.
If Job is to leave:
 - Processes its departure, returns nothing
If it's to move:
 - Returns the appropriate MoveEvent or OverflowEvent (determined by is_overflow flag)
"""
function process_transfer_from(state::DiscreteState, server::Server, job::Job, is_overflow::Bool)
    ind = is_overflow ? get_overflow_index_from(server) : get_move_index_from(server)
    if ind == 0
        leave_job(state, job)
        return nothing
    else
        state.moving_jobs_num += 1
        return is_overflow ? OverflowEvent(get_transfer_from_index(state, ind), job) : MoveEvent(get_transfer_from_index(state, ind), job)
    end 
end

"""
Handles adding a Job to a Server. 
If buffer is full:
 - Return OverflowEvent if moving to different buffer
 - Return nothing if leaving system
If buffer is empty:
 - Add Job to Server, return ServedEvent
Else:
 - Add Job to Server buffer, return nothing
"""
function add_job(state::DiscreteState, server::Server, job::Job)
    if isfull(server)
        return process_transfer_from(state, server, job, true)
    else
        ret = isempty(server) ? ServedEvent(server) : nothing
        enqueue!(server.buffer, job)
        return ret
    end
end


"""
For when a server finishes processing its job
"""
function process_job(server::Server)
    return dequeue!(server.buffer)
end

"""
    process_event(time::Float64, state::DiscreteState, event::ArrivalEvent)

Processes a Job's arrival:
 - Schedules the next Job's arrival
 - Attempts to add this job to a server
"""
function process_event(time::Float64, state::DiscreteState, event::ArrivalEvent)
    
    # # state.is_debug && println("Job #", state.last_job, " Arrived")
    state.last_job += 1
    state.jobs_in_system += 1
    new_timed_events = TimedEvent[]

    # Prepare next arrival
    push!(new_timed_events,generate_timed_event(time, state, ArrivalEvent(state.last_job)))

    server = state.servers[sample_probabilities(state.params.p_e, .0)]  # We will need to assert sum(p_e) = 1

    new_event = add_job(state, server, event.job)
    !isnothing(new_event) && push!(new_timed_events, generate_timed_event(time,state, new_event))
    # # If this is the only job on the server
    # state.number_in_system == 1 && push!(new_timed_events,TimedEvent(EndOfServiceEvent(), time + 1/μ))
    # Increase number in system
    
    return new_timed_events
end

"""
    process_event(time::Float64, state::DiscreteState, event::ServedEvent)

Handles a `Job`'s completion:
- Removes `Job` from Server
 - If buffer's not empty, create new `ServedEvent` for `Server`
- Handles `Job`'s subsequent path
 - If it moves to another `Server`, create `TransferEvent`
 - Else, do nothing
"""
function process_event(time::Float64, state::DiscreteState, event::ServedEvent)
    job = process_job(event.server)
    # # state.is_debug && println("Served Job #", job.unique_id, " at Server #", event.server.unique_id)
    new_timed_events = TimedEvent[]
    !isempty(event.server) && push!(new_timed_events, generate_timed_event(time, state, ServedEvent(event.server)))

    new_event = process_transfer_from(state, event.server, job, false)
    
    # Decide which server to send the job to, based on the event Vector
        # If sent out of system, return that event instead
    !isnothing(new_event) && append!(new_timed_events, [generate_timed_event(time, state, new_event)])
    # Decide whether to move normally or overflow
        # Return the appropriate event
    return new_timed_events
end


"""
    process_event(time::Float64, state::DiscreteState, event::TransferEvent)
    
Attempts to add a `Job` to a `Server`'s buffer, handles results
"""
function process_event(time::Float64, state::DiscreteState, event::TransferEvent)
    job = event.job
    state.moving_jobs_num -= 1
    server_to = event.transfer_to
    # # state.is_debug && println("Transferring Job #", job.unique_id, " to Server #", server_to.unique_id)
    new_event = add_job(state, server_to, job)
    return !isnothing(new_event) ? TimedEvent[generate_timed_event(time, state, new_event)] : TimedEvent[]
end

"""
    generate_timed_event(time::Float64, state::DiscreteState, event::TransferEvent)

Generates a `TimedEvent` for a `TransferEvent`. Its `time` is the current + global gamma distributed `η`
"""
function generate_timed_event(time::Float64, state::DiscreteState, event::TransferEvent)::TimedEvent
    return TimedEvent(event, time + generate_time(state.params.η, state.params.gamma_scv))
end

"""
    generate_timed_event(time::Float64, state::DiscreteState, event::ServedEvent)

Generates a `TimedEvent` for a `ServedEvent`. Its `time` is the current + gamma distributed `Server`'s rate
"""
function generate_timed_event(time::Float64, state::DiscreteState, event::ServedEvent)::TimedEvent
    return TimedEvent(event, time + generate_time(event.server.service_rate, state.params.gamma_scv))
end

"""
    generate_timed_event(time::Float64, state::DiscreteState, event::ArrivalEvent)

Generates a `TimedEvent` for an `ArrivalEvent`. Its `time` is the current + global arrival rate (will not be constant)
"""
function generate_timed_event(time::Float64, state::DiscreteState, event::ArrivalEvent)::TimedEvent
    return TimedEvent(event, time + generate_time(state.params.λ, state.params.gamma_scv))
end

"""
A convenience function to make a Gamma distribution with desired rate (inverse of shape) and SCV.
"""
function rate_scv_gamma(desired_rate::Float64, desired_scv::Float64)
    return Gamma(1/desired_scv, desired_scv/desired_rate)
end

"""
Generates a time based on mean rate μ and SCV gamma_scv
"""
function generate_time(μ::Float64, gamma_scv::Float64)
    rand(rate_scv_gamma(μ, gamma_scv))
end