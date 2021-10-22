using Distributions, Random, Plots

include("simulate.jl")

Random.seed!(0)

λ = 1.8
μ = 2.0
 
mutable struct QueueState <: State
    number_in_system::Int # If ≥ 1 then server is busy, If = 0 server is idle.
end

struct ArrivalEvent <: Event end
struct EndOfServiceEvent <: Event end

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

# Process an end of service event 
function process_event(time::Float64, state::State, ::EndOfServiceEvent)
    # Release a customer from the system
    state.number_in_system -= 1 
    @assert state.number_in_system ≥ 0
    return state.number_in_system ≥ 1 ? [TimedEvent(EndOfServiceEvent(), time + 1/μ)] : TimedEvent[]
end

simulate(QueueState(0), TimedEvent(ArrivalEvent(),0.0), log_times = [5.3,7.5])

time_traj, queue_traj = Float64[], Int[]

function record_trajectory(time::Float64, state::QueueState) 
    push!(time_traj, time)
    push!(queue_traj, state.number_in_system)
    return nothing
end

simulate(QueueState(0), TimedEvent(ArrivalEvent(),0.0), max_time = 100.0, callback = record_trajectory)

plot(stich_steps(time_traj, queue_traj)... ,
             label = false, xlabel = "Time", ylabel = "Queue size (number in system)" )

λ = 1.8
μ = 2.0

prev_time = 0.0
prev_state = 0
integral = 0.0

function add_to_integral(time::Float64, state::QueueState) 
    # Make sure to use the variables above
    global prev_time, prev_state, integral

    diff = time - prev_time
    integral += prev_state * diff
    prev_time = time
    prev_state = state.number_in_system

    return nothing
end

simulate(QueueState(0), TimedEvent(ArrivalEvent(),0.0), max_time = 10.0^6, callback = add_to_integral)
println("Simulated mean queue length: ", integral / 10^6 )

ρ = λ / μ
md1_theory = ρ/(1-ρ)*(2-ρ)/2
println("Theoretical mean queue length: ", md1_theory)