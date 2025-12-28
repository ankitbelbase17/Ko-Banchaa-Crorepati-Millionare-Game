#!/usr/bin/env python3
"""
Hunyuan3D-2.1 Docker Pipeline Runner
Generates 3D mesh (.glb) from input image using dockerized Hunyuan3D-2.1
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


class Hunyuan3DRunner:
    def __init__(self, docker_image="your_username/hunyuan3d:latest", data_dir="~/hunyuan_data"):
        """
        Initialize the Hunyuan3D pipeline runner
        
        Args:
            docker_image: Docker image name (e.g., 'username/hunyuan3d:latest')
            data_dir: Directory to store input/output files
        """
        self.docker_image = docker_image
        self.data_dir = Path(data_dir).expanduser()
        
    def check_docker_image_exists(self):
        """Check if the Docker image exists locally or can be pulled"""
        print(f"Checking Docker image: {self.docker_image}")
        
        # Check if image exists locally
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                print(f"✓ Docker image found locally: {self.docker_image}")
                return True
            else:
                print(f"⚠ Image not found locally. Attempting to pull from Docker Hub...")
                return self.pull_docker_image()
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Error checking Docker image: {e}")
            return False
    
    def pull_docker_image(self):
        """Pull Docker image from Docker Hub"""
        try:
            print(f"Pulling {self.docker_image} (this may take a while)...")
            result = subprocess.run(
                ["docker", "pull", self.docker_image],
                capture_output=False,
                text=True,
                check=True
            )
            print(f"✓ Successfully pulled {self.docker_image}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to pull Docker image: {e}")
            print(f"Please check:")
            print(f"  1. Image name is correct: {self.docker_image}")
            print(f"  2. Image exists on Docker Hub")
            print(f"  3. You're logged in (run: docker login)")
            return False
        
    def setup_data_directory(self):
        """Create data directory if it doesn't exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Data directory ready: {self.data_dir}")
        
    def copy_input_image(self, input_image_path):
        """Copy input image to data directory"""
        input_path = Path(input_image_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
        target_path = self.data_dir / "input.png"
        
        # Copy file
        import shutil
        shutil.copy2(input_path, target_path)
        print(f"✓ Input image copied to: {target_path}")
        return target_path
    
    def check_docker_available(self):
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Docker available: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Docker not found. Please install Docker first.")
            return False
    
    def check_gpu_available(self):
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ NVIDIA GPU detected")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ Warning: No NVIDIA GPU detected. Pipeline may fail or run very slowly.")
            return False
    
    def generate_shape(self, verbose=True):
        """
        Run the Docker container to generate 3D shape
        
        Args:
            verbose: Print detailed output
            
        Returns:
            Path to output .glb file
        """
        output_file = self.data_dir / "output_shape.glb"
        
        # Python script to run inside container
        python_script = """
import sys
sys.path.insert(0, './hy3dshape')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

print('Loading Hunyuan3D pipeline...')
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')

print('Generating mesh from /data/input.png...')
mesh = pipeline(image='/data/input.png')[0]

print('Saving to /data/output_shape.glb...')
mesh.export('/data/output_shape.glb')

print('✓ Shape generation complete!')
"""
        
        # Docker run command
        docker_cmd = [
            "docker", "run",
            "--gpus", "all",
            "--rm",  # Remove container after execution
            "-v", f"{self.data_dir.absolute()}:/data",
            "-w", "/workspace/Hunyuan3D-2.1",
            self.docker_image,
            "python3", "-c", python_script
        ]
        
        print(f"\n{'='*60}")
        print("Starting Hunyuan3D shape generation...")
        print(f"{'='*60}\n")
        
        try:
            # Run docker command
            result = subprocess.run(
                docker_cmd,
                capture_output=not verbose,
                text=True,
                check=True
            )
            
            if verbose and result.stdout:
                print(result.stdout)
            
            # Check if output file was created
            if output_file.exists():
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS! Output saved to: {output_file}")
                print(f"{'='*60}\n")
                return output_file
            else:
                raise RuntimeError("Output file was not created")
                
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error running Docker container:")
            if e.stderr:
                print(e.stderr)
            raise
    
    def run(self, input_image_path, verbose=True):
        """
        Complete pipeline: setup, copy image, generate shape
        
        Args:
            input_image_path: Path to input image
            verbose: Print detailed output
            
        Returns:
            Path to output .glb file
        """
        print("\n🚀 Hunyuan3D-2.1 Pipeline Starting...\n")
        
        # Pre-flight checks
        print("Step 1: Checking Docker...")
        if not self.check_docker_available():
            sys.exit(1)
        
        print("\nStep 2: Checking GPU...")
        self.check_gpu_available()
        
        print("\nStep 3: Checking Docker image...")
        if not self.check_docker_image_exists():
            sys.exit(1)
        
        print("\nStep 4: Setting up data directory...")
        self.setup_data_directory()
        
        print("\nStep 5: Copying input image...")
        self.copy_input_image(input_image_path)
        
        print("\nStep 6: Generating 3D shape...")
        output_file = self.generate_shape(verbose=verbose)
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D mesh from image using Hunyuan3D-2.1 Docker pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage (uses default Docker image name)
  python hunyuan3d_runner.py image.png
  
  # Specify your Docker Hub image
  python hunyuan3d_runner.py image.png -d myusername/hunyuan3d:latest
  
  # Custom output directory
  python hunyuan3d_runner.py image.png -o ./outputs
  
  # Quiet mode
  python hunyuan3d_runner.py image.png -q
        """
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "-d", "--docker-image",
        type=str,
        default="your_username/hunyuan3d:latest",
        help="Docker image name (default: your_username/hunyuan3d:latest)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="~/hunyuan_data",
        help="Output directory for generated files (default: ~/hunyuan_data)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run pre-flight checks only (don't generate mesh)"
    )
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = Hunyuan3DRunner(
        docker_image=args.docker_image,
        data_dir=args.output_dir
    )
    
    # Test mode - just run checks
    if args.test:
        print("\n🧪 Running pre-flight checks only...\n")
        print("Step 1: Checking Docker...")
        docker_ok = runner.check_docker_available()
        
        print("\nStep 2: Checking GPU...")
        runner.check_gpu_available()
        
        print("\nStep 3: Checking Docker image...")
        image_ok = runner.check_docker_image_exists()
        
        print("\nStep 4: Setting up data directory...")
        runner.setup_data_directory()
        
        print("\nStep 5: Checking input image...")
        input_path = Path(args.input_image)
        if input_path.exists():
            print(f"✓ Input image found: {input_path}")
        else:
            print(f"✗ Input image not found: {input_path}")
        
        print("\n" + "="*60)
        if docker_ok and image_ok:
            print("✅ All checks passed! Ready to generate.")
        else:
            print("❌ Some checks failed. Please fix the issues above.")
        print("="*60)
        return
    
    # Normal mode - run full pipeline
    try:
        output_file = runner.run(
            input_image_path=args.input_image,
            verbose=not args.quiet
        )
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📦 Output: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()