using Random, StatsBase, Parameters, LinearAlgebra
import Base: isempty
include("simulate.jl")

struct Job
    unique_id::Int64
end

struct Server
    buffer::Queue{Job}           # First element in buffer is currently served job
    max_buffer_size::Int         # Buffer size before overflow
    overflow_vector::Vector{Int} # Probabilities of where an overflown job will go
    move_vector::Vector{Int}     # Probabilities of where a finished job will go
    service_rate::Float64        # Mean service rate of this server
end

function Server(max_buffer_size::Int,       
    overflow_vector::Vector{Int}, 
    move_vector::Vector{Int},  
    service_rate::Float64)
    return Server(Queue{Job}(), max_buffer_size, overflow_vector, move_vector, service_rate)
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


struct DiscreteState <: State
    servers::Vector{Server}     # Enumerated list of servers
    left_jobs::Vector{Job}      # Enumerated list of jobs that have left the system
    moving_jobs::Vector{Job}    # Enumerated list of jobs that are moving across servers
    last_job::Int64             # Index of last-added job
    params::NetworkParameters   # Hold our params in the state (is this a good idea? we'll find out)
end

"""
    DiscreteState(params::NetworkParameters)

Creates a DiscreteState, populates it with `servers` and `params` based on the given. First `job` index is 1, and all `job` storages start empty
"""
function DiscreteState(params::NetworkParameters)
    servers = [Server(params.K[i], params.Q[i], params.P[i], params.μ_vector[i]) for i in 1:params.L]
    return DiscreteState(servers, Vector{Job}(), Vector{Job}(), 1, params)
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
function sample_probabilities(transfer_probs::Vector{Float64})
    return sample(vcat(0, 1:length(transfer_probs)), Weights(vcat(1-sum(transfer_probs), transfer_probs)), 1)
end

"""
Returns server index a job should overflow to, or 0 if job should leave
"""
function get_overflow_index_from(server::Server)::Int
    return sample_probabilities(server.overflow_vector)
end

"""
Returns server index a job should move to, or 0 if job should leave
"""
function get_move_index_from(server::Server)::Int
    return sample_probabilities(server.move_vector)
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
    push!(state.left_jobs, job)
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
        enqueue!(buffer, job)
        isempty(server) && return ServedEvent(server)
        return nothing
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
    # Increase number in system
    state.last_job += 1
    new_timed_events = TimedEvent[]

    # Prepare next arrival
    push!(new_timed_events,generate_timed_event(time, state, ArrivalEvent(state.last_job)))

    server = state.servers[sample_probabilities(state.params.p_e)]  # We will need to assert sum(p_e) = 1

    new_event = add_job(state, server, event.job)
    !isnothing(new_event) && push!(new_timed_events, generate_timed_event(time,state, new_event))
    # # If this is the only job on the server
    # state.number_in_system == 1 && push!(new_timed_events,TimedEvent(EndOfServiceEvent(), time + 1/μ))
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
    server_to = event.transfer_to
    new_event = add_job(state, server_to, job)
    return !isnothing(new_event) ? TimedEvent[generate_timed_event(time, state, new_event)] : TimedEvent[]
end

"""
    generate_timed_event(time::Float64, state::DiscreteState, event::TransferEvent)

Generates a `TimedEvent` for a `TransferEvent`. Its `time` is the current + global gamma distributed `η`
"""
function generate_timed_event(time::Float64, state::DiscreteState, event::TransferEvent)::TimedEvent
    return TimedEvent(time + generate_time(state.params.η, state.params.gamma_scv), event)
end

"""
    generate_timed_event(time::Float64, state::DiscreteState, event::ServedEvent)

Generates a `TimedEvent` for a `ServedEvent`. Its `time` is the current + gamma distributed `Server`'s rate
"""
function generate_timed_event(time::Float64, state::DiscreteState, event::ServedEvent)::TimedEvent
    return TimedEvent(time + generate_time(event.server.service_rate, state.params.gamma_scv), event)
end

"""
    generate_timed_event(time::Float64, state::DiscreteState, event::ArrivalEvent)

Generates a `TimedEvent` for an `ArrivalEvent`. Its `time` is the current + global arrival rate (will not be constant)
"""
function generate_timed_event(time::Float64, state::DiscreteState, event::ArrivalEvent)::TimedEvent
    return TimedEvent(time + generate_time(state.params.λ, state.params.gamma_scv), event)
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