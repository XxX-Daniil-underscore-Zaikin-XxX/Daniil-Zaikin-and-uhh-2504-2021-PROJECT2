include("simulate.jl")
include("serverbuffer.jl")

scenario1 = NetworkParameters(  L=3, 
                                gamma_scv = 3.0, 
                                λ = 1, 
                                η = 4.0, 
                                μ_vector = ones(3),
                                P = [0 1.0 0;
                                    0 0 1.0;
                                    0 0 0],
                                Q = zeros(3,3),
                                p_e = [1.0, 0, 0],
                                K = fill(5,3))
@show scenario1
state = DiscreteState(scenario1)
simulate(state, generate_timed_event(.0, state, ArrivalEvent(state.last_job)), max_time = 20.0)