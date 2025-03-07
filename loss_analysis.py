import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from Losses import blue_loss, custom_color_loss, EPSLoss, TAPLoss, SALoss

# Suppress warnings
warnings.filterwarnings('ignore')

class LossAnalyzer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512))
        ])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize metrics
        self.lpips = LPIPS(net_type='alex').to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        
        # Suppress LPIPS warning by setting torch.load default
        torch.set_warn_always(False)
        
        # Initialize all loss functions
        self.eps_loss = EPSLoss().to(self.device)
        self.tap_loss = TAPLoss().to(self.device)
        self.sa_loss = SALoss().to(self.device)
        
    def load_images(self, style_name):
        """Load images for a specific style across all loss types including blue_loss."""
        loss_types = ['Blue', 'Color', 'EPS', 'TAP', 'SA']  # Added Blue loss
        images = {}
        
        for loss_type in loss_types:
            # For blue loss, look in the original output directory
            if loss_type == 'Blue':
                path = f"outputs/{style_name}.png"
            else:
                path = f"outputs_{loss_type}/{style_name}.png"
                
            if os.path.exists(path):
                img = Image.open(path)
                images[loss_type] = self.transform(img).unsqueeze(0).to(self.device)
        
        return images

    def compute_metrics(self, images):
        """Compute various metrics between images."""
        metrics = {
            'ssim': {},
            'psnr': {},
            'lpips': {},
            'color_distribution': {},
            'edge_intensity': {},
            'loss_values': {}  # New metric to store actual loss values
        }
        
        # Reference image (Blue loss as baseline)
        ref_img = images['BLUE']  # Now using uppercase key
        
        for loss_type, img in images.items():
            if loss_type != 'BLUE':  # Compare with uppercase key
                # Structural Similarity
                metrics['ssim'][loss_type] = self.ssim(img, ref_img).item()
                
                # Peak Signal-to-Noise Ratio
                metrics['psnr'][loss_type] = self.psnr(img, ref_img).item()
                
                # Learned Perceptual Image Patch Similarity
                metrics['lpips'][loss_type] = self.lpips(img, ref_img).item()
            
            # Color Distribution Analysis
            metrics['color_distribution'][loss_type] = {
                'mean': img.mean((2, 3)).squeeze().cpu().numpy(),
                'std': img.std((2, 3)).squeeze().cpu().numpy()
            }
            
            # Edge Intensity
            edges = self.compute_edge_intensity(img)
            metrics['edge_intensity'][loss_type] = edges.mean().item()
            
            # Compute actual loss values for each loss function
            metrics['loss_values'][loss_type] = {
                'blue_loss': blue_loss(img).item(),
                'custom_color': custom_color_loss(img).item(),
                'eps_loss': self.eps_loss(img).item(),
                'tap_loss': self.tap_loss(img).item(),
                'sa_loss': self.sa_loss(img).item()
            }
        
        return metrics

    def compute_edge_intensity(self, img):
        """Compute edge intensity using Sobel filters."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                             dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_x.transpose(2, 3)
        
        if torch.cuda.is_available():
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
            
        edges_x = torch.nn.functional.conv2d(img, sobel_x.repeat(3,1,1,1), groups=3, padding=1)
        edges_y = torch.nn.functional.conv2d(img, sobel_y.repeat(3,1,1,1), groups=3, padding=1)
        
        return torch.sqrt(edges_x.pow(2) + edges_y.pow(2))

    def plot_comparisons(self, style_name, images, metrics, save_path):
        """Create comparative visualizations."""
        sns.set_style("whitegrid")
        
        # Create a larger figure to accommodate the new plot
        fig = plt.figure(figsize=(25, 20))
        fig.suptitle(f'Loss Function Comparison for {style_name} Style', fontsize=16)
        
        # Original plots
        ax1 = plt.subplot(3, 2, 1)
        self.plot_images(images, ax1)
        
        ax2 = plt.subplot(3, 2, 2)
        self.plot_similarity_metrics(metrics, ax2)
        
        ax3 = plt.subplot(3, 2, 3)
        self.plot_color_distribution(metrics, ax3)
        
        ax4 = plt.subplot(3, 2, 4)
        self.plot_edge_intensity(metrics, ax4)
        
        # New plot for loss values comparison
        ax5 = plt.subplot(3, 2, (5, 6))
        self.plot_loss_values(metrics, ax5)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_images(self, images, ax):
        """Plot image grid."""
        grid_size = int(np.ceil(np.sqrt(len(images))))
        for i, (loss_type, img) in enumerate(images.items()):
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(img.squeeze().permute(1,2,0).cpu())
            plt.title(loss_type, fontsize=10)
            plt.axis('off')

    def plot_similarity_metrics(self, metrics, ax):
        """Plot similarity metrics comparison."""
        loss_types = list(metrics['ssim'].keys())
        x = np.arange(len(loss_types))
        width = 0.25
        
        colors = sns.color_palette("husl", 3)
        
        ax.bar(x - width, [metrics['ssim'][lt] for lt in loss_types], width, label='SSIM', color=colors[0])
        ax.bar(x, [metrics['psnr'][lt] for lt in loss_types], width, label='PSNR', color=colors[1])
        ax.bar(x + width, [metrics['lpips'][lt] for lt in loss_types], width, label='LPIPS', color=colors[2])
        
        ax.set_xticks(x)
        ax.set_xticklabels(loss_types, rotation=45)
        ax.set_title('Similarity Metrics', pad=20)
        ax.legend()

    def plot_color_distribution(self, metrics, ax):
        """Plot color distribution comparison."""
        colors = sns.color_palette("husl", len(metrics['color_distribution']))
        
        for i, (loss_type, dist) in enumerate(metrics['color_distribution'].items()):
            ax.bar(np.arange(3) + 0.2*i,
                  dist['mean'],
                  yerr=dist['std'],
                  width=0.2,
                  label=loss_type,
                  color=colors[i])
        
        ax.set_xticks(np.arange(3) + 0.3)
        ax.set_xticklabels(['R', 'G', 'B'])
        ax.set_title('Color Distribution', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def plot_edge_intensity(self, metrics, ax):
        """Plot edge intensity comparison."""
        # Create DataFrame for seaborn
        data = {
            'Loss Type': list(metrics['edge_intensity'].keys()),
            'Edge Intensity': list(metrics['edge_intensity'].values())
        }
        
        # Plot using seaborn with proper parameters
        sns.barplot(
            data=data,
            x='Loss Type',
            y='Edge Intensity',
            hue='Loss Type',
            legend=False,
            ax=ax
        )
        
        # Customize the plot
        ax.set_title('Edge Intensity', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='x', labelrotation=45)

    def plot_loss_values(self, metrics, ax):
        """Plot how well each image optimizes each loss objective."""
        loss_types = list(metrics['loss_values'].keys())
        loss_functions = ['blue_loss', 'custom_color', 'eps_loss', 'tap_loss', 'sa_loss']
        
        # Create a normalized comparison matrix
        comparison_data = {}
        for loss_fn in loss_functions:
            # Get values for this loss function across all images
            values = [metrics['loss_values'][lt][loss_fn] for lt in loss_types]
            # Normalize values to [0, 1] range for fair comparison
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized = [1.0 for _ in values]
            comparison_data[loss_fn] = normalized
        
        # Plot normalized values
        x = np.arange(len(loss_types))
        width = 0.15
        multiplier = 0
        
        # Create bar plot
        for loss_fn in loss_functions:
            offset = width * multiplier
            values = comparison_data[loss_fn]
            ax.bar(x + offset, values, width, label=f'{loss_fn} objective')
            multiplier += 1
        
        # Customize plot
        ax.set_title('Loss Objectives Comparison (Lower is Better)')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(loss_types, rotation=45)
        ax.set_ylabel('Normalized Loss Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add text explanation
        ax.text(-0.2, -0.3, 
                "Each bar shows how well an image optimizes each loss objective.\n" +
                "Lower values indicate better optimization of that objective.",
                transform=ax.transAxes, fontsize=8, style='italic')

def analyze_prompt_outputs(prompt_dir):
    """Analyze all styles and losses for a specific prompt directory."""
    analyzer = LossAnalyzer()
    
    # Get the latest timestamp directory
    timestamp_dirs = [d for d in os.listdir(prompt_dir) if os.path.isdir(os.path.join(prompt_dir, d))]
    if not timestamp_dirs:
        print(f"No timestamp directories found in {prompt_dir}")
        return
    latest_timestamp = sorted(timestamp_dirs)[-1]
    generation_path = os.path.join(prompt_dir, latest_timestamp)
    
    # Create analysis directory
    analysis_dir = os.path.join(prompt_dir, latest_timestamp, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Get all style directories
    style_dirs = [d for d in os.listdir(generation_path) 
                 if os.path.isdir(os.path.join(generation_path, d)) and d != 'analysis']
    
    with open(os.path.join(analysis_dir, 'analysis_summary.txt'), 'w') as f:
        for style_name in style_dirs:
            print(f"\nAnalyzing {style_name}...")
            print("="*100)
            f.write(f"\nAnalysis for {style_name}:\n")
            f.write("="*100 + "\n")
            
            # Load images for this style
            images = {}
            style_path = os.path.join(generation_path, style_name)
            
            # Load all images and store with uppercase keys for consistency
            for loss_type in ['blue', 'color', 'eps', 'tap', 'sa']:
                img_path = os.path.join(style_path, loss_type, f"{style_name}.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    images[loss_type.upper()] = analyzer.transform(img).unsqueeze(0).to(analyzer.device)
            
            if not images:
                print(f"No images found for {style_name}")
                continue
                
            if 'BLUE' not in images:
                print(f"No blue loss image found for {style_name}, using first available image as reference")
                # Use the first available image as reference if blue loss image is not found
                ref_key = list(images.keys())[0]
                images['BLUE'] = images[ref_key]
            
            try:
                metrics = analyzer.compute_metrics(images)
                
                # Save analysis plot
                analyzer.plot_comparisons(style_name, images, metrics, 
                                       save_path=os.path.join(analysis_dir, f'analysis_{style_name}.png'))
                
                # Write detailed analysis
                f.write("\nComparison with Blue Loss baseline:\n")
                for loss_type in ['COLOR', 'EPS', 'TAP', 'SA']:
                    if loss_type in metrics['ssim']:
                        f.write(f"\n{loss_type} Loss:\n")
                        f.write(f"SSIM: {metrics['ssim'][loss_type]:.3f}\n")
                        f.write(f"PSNR: {metrics['psnr'][loss_type]:.3f}\n")
                        f.write(f"LPIPS: {metrics['lpips'][loss_type]:.3f}\n")
                
                f.write("\nLoss Values:\n")
                for loss_type, values in metrics['loss_values'].items():
                    f.write(f"\n{loss_type}:\n")
                    for loss_name, value in values.items():
                        f.write(f"{loss_name}: {value:.3f}\n")
                
                print(f"Analysis complete for {style_name}")
                print(f"Results saved in {os.path.join(analysis_dir, f'analysis_{style_name}.png')}")
                
            except Exception as e:
                print(f"Error analyzing {style_name}: {str(e)}")
                continue

def main():
    NUM_DASHES = 100
    print("Starting analysis...")
    print("="*NUM_DASHES)
    
    # Look for prompt directories in the generated_images folder
    base_dir = "generated_images"
    if not os.path.exists(base_dir):
        print(f"No generated images found in {base_dir}")
        return
    
    prompt_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for prompt_dir in prompt_dirs:
        print(f"\nAnalyzing outputs for prompt: {prompt_dir}")
        print("="*NUM_DASHES)
        analyze_prompt_outputs(os.path.join(base_dir, prompt_dir))

if __name__ == "__main__":
    main() 