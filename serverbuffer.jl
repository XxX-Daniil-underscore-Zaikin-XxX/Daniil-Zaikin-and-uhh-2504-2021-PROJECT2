using Random, StatsBase, Parameters, LinearAlgebra
include("simulate.jl")

struct Job
    unique_id::Int64
end

struct Server
    buffer::Queue{Job}         # First element in buffer is currently served job
    max_buffer_size::Int       # Buffer size before overflow
    overflow_vector::Vector{Int} # Probabilities of where an overflown job will go
    move_vector::Vector{Int}   # Probabilities of where a finished job will go
    service_rate::Float64      # Mean service rate of this server
end

struct ArrivalEvent <: Event end

struct TransferEvent <: Event
    transfer_to::Server
    job::Job
end

struct OverflowEvent <: TransferEvent end
struct MoveEvent <: TransferEvent end

struct LeaveEvent <: Event
    job::Job
end

struct BufferEvent <: Event 

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
    params::NetworkParameters   # Hold our params in the state (is this a good idea? we'll find out)
end

"""
Pull out the important info from our state - we don't need our params
"""
function copy_state_items(state::DiscreteState)
    return deepcopy(state.servers), deepcopy(state.left_jobs)
end

"""
Returns whether a server's buffer is full
"""
function isfull(server::Server)
    return length(server.buffer) > server.max_buffer_size
end

"""
Returns whether a server has no current job and empty buffer
"""
function isempty(server::Server)
    return isempty(server.buffer)
end

"""
Returns index of server a job should transfer to, or 0 if the job should leave
"""
function sample_server(transfer_probs::Vector{Float16})
    return sample(vcat(0, 1:length(transfer_probs)), Weights(vcat(1-sum(transfer_probs), transfer_probs)), 1)
end

"""
Returns server index a job should overflow to, or 0 if job should leave
"""
function get_overflow_index_from(server::Server)::Int
    return sample_server(server.overflow_vector)
end

"""
Returns server index a job should move to, or 0 if job should leave
"""
function get_move_index_from(server::Server)::Int
    return sample_server(server.move_vector)
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
function process_transfer(state::DiscreteState, server::Server, job::Job, is_overflow::Bool)
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
    - Add job to Server buffer, return nothing
"""
function add_job(state::DiscreteState, server::Server, job::Job)
    if isfull(server)
        return process_transfer(state, server, job, true)
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

# Process an arrival event
function process_event(time::Float64, state::State, ::ArrivalEvent)
    # Increase number in system
    state.number_in_system += 1
    new_timed_events = TimedEvent[]

    # Prepare next arrival
    push!(new_timed_events,TimedEvent(ArrivalEvent(),time + rand(Exponential(1/λ))))

    # If this is the only job on the server
    state.number_in_system == 1 && push!(new_timed_events,TimedEvent(EndOfServiceEvent(), time + 1/μ))
    return new_timed_events
end

# When a job finishes being served
function process_event(time::Float64, state::DiscreteState, event::ServedEvent)
    job = process_job(event.server)

    new_timed_events = TimedEvent[]

    new_event = process_transfer(state, event.server, job, false)
    # Decide which server to send the job to, based on the event Vector
        # If sent out of system, return that event instead
    !isnothing(new_event) && append!(new_timed_events, [generate_timed_event(time, state, new_event)])
    # Decide whether to move normally or overflow
        # Return the appropriate event
    return new_timed_events
end

#!!!BAD EVENT - LEAVING HAPPENS IMMEDIATELY!!!
function process_event(time::Float64, state::State, event::LeaveEvent)
    job = event.job
    # Add the job to our 'outside' jobs
    # Return no events
end

function process_event(time::Float64, state::DiscreteState, event::MoveEvent)
    job = event.job
    server_to = event.transfer_to
    new_event = add_job(state, server_to, job)
    return !isnothing(new_event) ? TimedEvent[generate_timed_event(time, state, new_event)] : TimedEvent[]
end

function process_event(time::Float64, state::State, event::OverflowEvent)
    job = event.job
    server_to = event.transfer_to
    # Add job to server, return appropriate event (overflow, move, serve, nothing)
end

function generate_timed_event(time::Fload64, state::DiscreteState, event::TransferEvent)
    return TimedEvent(time + generate_time(state.params.η, state.params.gamma_scv), event)
end

function generate_timed_event(time::Fload64, state::DiscreteState, event::ServedEvent)
    return TimedEvent(time + generate_time(event.server.service_rate, state.params.gamma_scv), event)
end

function generate_timed_event(time::Float64, state::DiscreteState, event::ArrivalEvent)
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