using Profile, Plots

include("simulate.jl")
include("serverbuffer.jl")
include("projectscenarios.jl")

function do_experiments(;max_time = 10.0^5, λ = [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3], params_all = [scenario1, scenario2, scenario3, scenario4, scenario5])

    # 10^5 because it'll take too long to do 10^7. Each sim with the latter time runs for ~30 secs, which I'd say is acceptable, but not acceptable enough for me to test it half an hour before deadline
    mean_sum = 0
    orbit_sum = 0
    last_time = .0

    # Adds sum of jobs in system to running total
    function record_total_sum(time::Float64, state::DiscreteState)
        mean_sum += state.jobs_in_system*(time-last_time)
    end

    # Adds sum of orbiting jobs to running total
    function record_total_orbit(time::Float64, state::DiscreteState)
        orbit_sum += state.moving_jobs_num*(time-last_time)
    end

    # Wraps all our recordings
    function record_all(time::Float64, state::DiscreteState)
        record_total_sum(time, state)
        record_total_orbit(time, state)
        last_time = time
    end

    current_scenario = 1

    #Iterate through all our records, give back a chart
    for params in params_all
        mean_vec = []
        orbit_vec = []
        for i in λ
            mean_sum = 0
            orbit_sum = 0
            last_time = .0
            new_params = NetworkParameters(params.L, params.gamma_scv, i, params.η, params.μ_vector, params.P, params.Q, params.p_e, params.K)
            state = DiscreteState(new_params)
            @time simulate(state, generate_timed_event(.0, state, ArrivalEvent(state.last_job)), max_time=max_time, callback=record_all)
            push!(mean_vec, mean_sum/max_time)
            push!(orbit_vec, orbit_sum/mean_sum)
        end
        print(mean_vec)
        print(orbit_vec)
        mean_plot = plot(λ, mean_vec, title="Scenario $current_scenario: Mean Jobs in System", xlabel="λ", ylabel="Total Items",legend=false)
        orbit_plot = plot(λ, mean_vec, title="Scenario $current_scenario: Mean Orbiting Jobs in System", xlabel="λ", ylabel="Orbiting Items",legend=false)
        display(plot(mean_plot, orbit_plot, layout = (2, 1),legend=false, size = (600, 900)))
        current_scenario += 1
    end
end

do_experiments()