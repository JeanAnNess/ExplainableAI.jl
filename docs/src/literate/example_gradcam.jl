# # [Example: GradCAM heatmap overlay](@id example-gradCAM)
# Calculating the GradCAM of your input image results in an explanation with drastically lower dimensionality compared to your input image. 
# We can visualize the Class-Activation-Mapping better by overlaying it over the input image. We can do so using 'heatmap_overlay' from 
# [VisionHeatmaps.jl](https://julia-xai.github.io/XAIDocs/VisionHeatmaps/stable/).
#
# This page showcases VisionHeatmaps' `heatmap_overlay` for GradCAM by demonstrating an examplary workflow
# which includes the preprocessing of the image via [ImageNetDataset.jl](https://github.com/adrhill/ImageNetDataset.jl).
#
# We start out by loading the VGG16 model:
using ExplainableAI
using VisionHeatmaps                        # displaying heatmap
using Flux          
using Metalhead                             # pre-trained vision models

model = VGG(16, pretrain=true).layers

# Next up an input image is required. In this example we load one via URL:
using ImageInTerminal, ImageShow            # display image         
using HTTP, FileIO                          # load image from URL  

url = HTTP.URI("https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg")
img = load(url)

# # Preprocessing using ImageNetDataset
# In order to make the image compatible with Flux we need to change the input to have the WHCN format. 
# For that we can use [ImageNetDataset.jl's](https://github.com/adrhill/ImageNetDataset.jl) `transform` 
# function to receive the WHC representation which we then reshape to WHCN.
using ImageNetDataset                       # reshape and display heatmap   

temp_name = tempname() * ".jpg"             # ImageNetDataset.transform() requires a file path
save(File(format"JPEG", temp_name), img)
tsf = CenterCropNormalize(; output_size=(224, 224))
input = transform(tsf, temp_name)           # transform image to WHC format
input = reshape(input, 224, 224, 3, :)      # reshape to WHCN format to make it Flux compatible

#md # !!! note "Temporary Files"
#md #
#md #     The file is saved to a temporary location. While it should be automatically deleted
#md #     when the Julia session ends, you can also manually delete it by calling `rm(temp_name)`
#md #     or by navigating to its location and deleting it manually.

# # Calculate explanation
# Using XAI methods we first create the GradCAM analyzer with which the explanation is calculated.
feature_layers = model.layers[1]
adaption_layers = model.layers[2]
analyzer = GradCAM(feature_layers, adaption_layers)
expl = analyze(input,analyzer)

# # Display heatmap overlay
# We can now overlay the heatmap over the input image using `heatmap_overlay`.
heatmap_overlay(expl, img; alpha = 0.6)

# # Optional: Save heatmap overlay
# heatmap_overlays are regular Images.jl images, they can be saved to a file using `save`.
overlay_img = heatmap_overlay(expl, img; alpha = 0.6)
save("heatmap_overlay.png", overlay_img)

# Close and delete temp file. This should be redundant, but it's good practice.
rm(temp_name)