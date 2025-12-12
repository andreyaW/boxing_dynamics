import gradio as gr
import os
from main import process_video

def run_and_return_file(uploaded_file):
    """
    uploaded_file is a filepath (type='filepath').
    We pass that to process_video() and return both:
      - the video for inline display
      - the same file for download
    """
    if uploaded_file is None:
        return None, None

    out_path = process_video(uploaded_file)

    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Expected output not found: {out_path}")

    # Return twice: once for Video preview, once for File download
    return out_path, out_path


with gr.Blocks(title="Boxing Dynamics Video Processor") as demo:
    gr.Markdown(
        """
        ## Boxing Dynamics Video Processor  
        Upload a video, let the model process it, watch the output directly, and download the result.
        """
    )

    with gr.Row():
        input_video = gr.File(
            label="Upload a video",
            file_count="single",
            type="filepath"
        )

    process_btn = gr.Button("Process Video")

    with gr.Row():
        output_video = gr.Video(label="Processed Video Preview")
    
    with gr.Row():
        download_file = gr.File(label="Download Processed Video")

    process_btn.click(
        fn=run_and_return_file,
        inputs=input_video,
        outputs=[output_video, download_file]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)