import sys
import os

# Ensure the root directory is in the path so imports work
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.workflow.workflow import graph
from IPython.display import Image, display

def generate_diagram():
    print("Generating mermaid diagram string...")
    mermaid_code = graph.get_graph().draw_mermaid()
    
    with open("architecture_diagram.md", "w") as f:
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```\n")
    print("Mermaid diagram code saved to architecture_diagram.md")
    
    try:
        print("Generating and saving PNG image using draw_mermaid_png()...")
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("architecture_diagram.png", "wb") as f:
            f.write(png_bytes)
        
        # Using IPython.display as requested (though it only renders in Jupyter notebooks)
        img = Image(png_bytes)
        display(img)
        print("PNG image saved to architecture_diagram.png and passed to IPython.display.")
        
    except Exception as e:
        print(f"Failed to generate PNG (may require dependencies like graphviz or internet connection): {e}")

if __name__ == "__main__":
    generate_diagram()
