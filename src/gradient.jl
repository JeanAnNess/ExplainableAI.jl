function gradient_wrt_input(model, input, ns::AbstractNeuronSelector)
    output, back = Zygote.pullback(model, input)
    output_indices = ns(output)

    # Compute VJP w.r.t. full model output, selecting vector s.t. it masks output neurons
    v = zero(output)
    v[output_indices] .= 1
    grad = only(back(v))
    return grad, output, output_indices
end

"""
    Gradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
"""
struct Gradient{M} <: AbstractXAIMethod
    model::M
    Gradient(model) = new{typeof(model)}(model)
    Gradient(model::Chain) = new{typeof(model)}(Flux.testmode!(check_output_softmax(model)))
end

function (analyzer::Gradient)(input, ns::AbstractNeuronSelector)
    grad, output, output_indices = gradient_wrt_input(analyzer.model, input, ns)
    return Explanation(grad, output, output_indices, :Gradient, nothing)
end

"""
    InputTimesGradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
This gradient is then multiplied element-wise with the input.
"""
struct InputTimesGradient{M} <: AbstractXAIMethod
    model::M
    InputTimesGradient(model) = new{typeof(model)}(model)
    function InputTimesGradient(model::Chain)
        new{typeof(model)}(Flux.testmode!(check_output_softmax(model)))
    end
end

function (analyzer::InputTimesGradient)(input, ns::AbstractNeuronSelector)
    grad, output, output_indices = gradient_wrt_input(analyzer.model, input, ns)
    attr = input .* grad
    return Explanation(attr, output, output_indices, :InputTimesGradient, nothing)
end

"""
    SmoothGrad(analyzer, [n=50, std=0.1, rng=GLOBAL_RNG])
    SmoothGrad(analyzer, [n=50, distribution=Normal(0, σ²=0.01), rng=GLOBAL_RNG])

Analyze model by calculating a smoothed sensitivity map.
This is done by averaging sensitivity maps of a `Gradient` analyzer over random samples
in a neighborhood of the input, typically by adding Gaussian noise with mean 0.

# References
- $REF_SMILKOV_SMOOTHGRAD
"""
SmoothGrad(model, n=50, args...) = NoiseAugmentation(Gradient(model), n, args...)

"""
    IntegratedGradients(analyzer, [n=50])
    IntegratedGradients(analyzer, [n=50])

Analyze model by using the Integrated Gradients method.

# References
- $REF_SUNDARARAJAN_AXIOMATIC
"""
IntegratedGradients(model, n=50) = InterpolationAugmentation(Gradient(model), n)

"""
    GradCam

# 
""" 
struct GradCAM{C1,C2} <: AbstractXAIMethod
    feature_layers::C1
    adaptation_layers::C2
    GradCAM(C1,C2) = new{typeof(C1),typeof(C2)}(C1,C2)
end
function (analyzer::GradCAM)(input, ns::AbstractNeuronSelector)
    A = analyzer.feature_layers(input)    # Forward pass
    grad,output,output_indices=gradient_wrt_input(analyzer.adaptation_layers,A,ns)   # Backpropagation

    # Determine neuron importance αₖᶜ = 1/Z * ∑_i ∑_j ∂yᶜ / ∂A_ij^k
    αᶜ = sum(grad, dims=(1,2)) / (size(grad,1)*size(grad,2))  
    Lᶜ = max.(sum(αᶜ .* A, dims=3),0)
    
    return Explanation(Lᶜ, output, output_indices, :GradCAM, nothing)
end

function find_gradcam_target_layer(model)
    for layer in reverse(model.layers)
        params = Flux.params(layer)
        if !isempty(params) && length(size(params[1])) == 4  #reversed order: find first layer with 4D Output (WHCB)
            return layer
        end
    end
    error("No 4D Output found. GradCam cannot be called")
end
struct GradCAM2{M} <: AbstractXAIMethod
    model::M
    GradCAM2(model) = new{typeof(model)}(model)
    GradCAM2(model::Chain) = new{typeof(model)}(Flux.testmode!(check_output_softmax(model)))
end
function (analyzer::GradCAM2)(input, ns::AbstractNeuronSelector)::Explanation
    feature_layers = find_gradcam_target_layer(model)     
    activation=feature_layers(input)  # Forward pass
    grad,output,output_indices=gradient_wrt_input(analyzer.adaptation_layers,A,ns)   # Backpropagation

    # Determine neuron importance αₖᶜ = 1/Z * ∑_i ∑_j ∂yᶜ / ∂A_ij^k
    αᶜ = sum(grad, dims=(1,2)) / (size(grad,1)*size(grad,2)) 
    #ReLU since we are only interested on areas with positive impact on selected class 
    Lᶜ = max.(sum(αᶜ .* A, dims=3),0)

    return Explanation(Lᶜ, output, output_indices, :GradCAM2, nothing)
end