using DataStructures
import Base: isless

abstract type Event end
abstract type State end

# Captures an event and the time it takes place
struct TimedEvent
    event::Event
    time::Float64
end

# Comparison of two timed events - this will allow us to use them in a heap/priority-queue
isless(te1::TimedEvent, te2::TimedEvent) = te1.time < te2.time

"""
    new_timed_events = process_event(time, state, event)

Generate an array of 0 or more new `TimedEvent`s based on the current `event` and `state`.
"""
function process_event end # This defines a function with zero methods (to be added later)

# Generic events that we can always use

"""
    EndSimEvent()

Return an event that ends the simulation.
"""
struct EndSimEvent <: Event end

function process_event(time::Float64, state::State, es_event::EndSimEvent)
    println("Ending simulation at time $time.")
    return []
end

"""
    LogStateEvent()

Return an event that prints a log of the current simulation state.
"""
struct LogStateEvent <: Event end

function process_event(time::Float64, state::State, ls_event::LogStateEvent)
    println("Logging state at time $time.")
    println(state)
    return []
end

"""
The main simulation function gets an initial state and an initial event
that gets things going. Optional arguments are the maximal time for the
simulation, times for logging events, and a call-back function.
"""
function simulate(init_state::State, init_timed_event::TimedEvent
                    ; 
                    max_time::Float64 = 10.0, 
                    log_times::Vector{Float64} = Float64[],
                    callback = (time, state) -> nothing)

    # The event queue
    priority_queue = BinaryMinHeap{TimedEvent}()

    # Put the standard events in the queue
    push!(priority_queue, init_timed_event)
    push!(priority_queue, TimedEvent(EndSimEvent(), max_time))
    for log_time in log_times
        push!(priority_queue, TimedEvent(LogStateEvent(), log_time))
    end

    # initilize the state
    state = deepcopy(init_state)
    time = 0.0

    # Callback at simulation start
    callback(time, state)

    # The main discrete event simulation loop - SIMPLE!
    while true
        # Get the next event
        timed_event = pop!(priority_queue)

        # Advance the time
        time = timed_event.time

        # Act on the event
        new_timed_events = process_event(time, state, timed_event.event) 

        # If the event was an end of simulation then stop
        if timed_event.event isa EndSimEvent
            break 
        end

        # The event may spawn 0 or more events which we put in the priority queue 
        for nte in new_timed_events
            push!(priority_queue,nte)
        end

        # Callback for each simulation event
        callback(time, state)
    end
end
"""
This function is designed to stich_steps of a discrete event curve.
"""
function stich_steps(epochs, values)
    n = length(epochs)
    new_epochs  = [epochs[1]]
    new_values = [values[1]]
    for i in 2:n
        push!(new_epochs, epochs[i])
        push!(new_values, values[i-1])
        push!(new_epochs, epochs[i])
        push!(new_values, values[i])
    end
    return (new_epochs, new_values)
end;