import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import cv2
import numpy as np
import os
import bpy  # Blender for 3D conversion
from transformers import pipeline
from PIL import Image, ImageDraw

# Load LLM for better prompt generation
llm = pipeline("text-generation", model="gpt-3.5-turbo")

def generate_prompt(room_type, style, wall_color, furniture):
    prompt_input = f"Generate a detailed prompt for an AI model to create a {style} {room_type} with {wall_color} walls and containing {', '.join(furniture)}. Make it highly descriptive."
    prompt_output = llm(prompt_input, max_length=100)[0]['generated_text']
    return prompt_output

# Automatically generate a sketch layout
def generate_sketch():
    sketch = Image.new("L", (256, 256), 255)
    draw = ImageDraw.Draw(sketch)
    draw.rectangle([50, 50, 200, 200], outline=0, width=5)
    draw.line([50, 125, 200, 125], fill=0, width=3)
    draw.line([125, 50, 125, 200], fill=0, width=3)
    sketch_path = "generated_sketch.png"
    sketch.save(sketch_path)
    return sketch_path

# Load Stable Diffusion Model
def generate_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe(prompt).images[0]
    image_path = "generated_design.png"
    image.save(image_path)
    return image_path

# Function to estimate depth (Using MiDaS)
def estimate_depth(image_path):
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384)) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        depth_map = model(img)
    depth_map = depth_map.squeeze().numpy()
    depth_image_path = "depth_map.png"
    cv2.imwrite(depth_image_path, (depth_map * 255).astype(np.uint8))
    return depth_image_path

# Convert depth map to 3D model using Blender
def depth_to_3d(depth_map_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()  # Clear scene
    
    bpy.ops.import_image.to_plane(files=[{"name": depth_map_path}])
    obj = bpy.context.selected_objects[0]
    obj.name = "Depth_Map_Plane"
    
    bpy.ops.export_scene.gltf(filepath="3d_model.glb")
    return "3d_model.glb"

# Streamlit UI for 3D Interior Design Generator
st.title("3D Interior Design Generator")

# Form for user inputs
with st.form("design_form"):
    room_type = st.selectbox("Select Room Type", ["Living Room", "Bedroom", "Kitchen", "Bathroom"])
    style = st.selectbox("Choose Style", ["Modern", "Minimalist", "Classic", "Industrial"])
    wall_color = st.color_picker("Select Wall Color")
    furniture = st.multiselect("Select Furniture", ["Sofa", "Bed", "Table", "Chair", "TV Unit", "Wardrobe"])
    submit = st.form_submit_button("Generate Design")

# Generate 3D design
if submit:
    st.write("Generating Room Layout Sketch...")
    sketch_path = generate_sketch()
    st.image(sketch_path, caption="Generated Sketch")
    
    prompt = generate_prompt(room_type, style, wall_color, furniture)
    st.write("Generating 3D Design with improved prompt...")
    
    image_path = generate_image(prompt)
    st.image(image_path, caption="Generated Interior Design")
    
    st.write("Estimating Depth Map...")
    depth_map_path = estimate_depth(image_path)
    st.image(depth_map_path, caption="Depth Map")
    
    st.write("Converting to 3D Model...")
    model_path = depth_to_3d(depth_map_path)
    
    st.write("Download your 3D Model:")
    st.download_button("Download 3D Model", model_path)

    # Embed Three.js viewer
    st.write("View your 3D Model:")
    three_js_viewer = """
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer();
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);

            const loader = new THREE.GLTFLoader();
            loader.load('3d_model.glb', function(gltf) {
                scene.add(gltf.scene);
                camera.position.z = 2;
                function animate() {
                    requestAnimationFrame(animate);
                    gltf.scene.rotation.y += 0.01;
                    renderer.render(scene, camera);
                }
                animate();
            });
        </script>
    </body>
    </html>
    """
    st.components.v1.html(three_js_viewer, height=600)

st.write("Built with Streamlit, Stable Diffusion, MiDaS, Blender, and Three.js")
