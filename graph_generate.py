import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_feature_analysis_graphs(csv_file_path):
    """
    Create 4 separate line graphs showing relationship between address accuracy and detected features.
    """
    
    # Create output directory for graphs
    output_dir = "feature_analysis_graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the columns we need (updated to v8)
    required_columns = [
        'address_ocr_v8_accuracy',  # Updated to v8
        'detected_angle',
        'detected_blur', 
        'detected_brightness',
        'detected_reflection'
    ]
    
    try:
        # Load only the required columns
        print("Loading data...")
        df = pd.read_csv(csv_file_path, usecols=required_columns)
        print(f"Loaded {len(df)} rows")
        
        # Remove rows with missing values in any of the required columns
        df_clean = df.dropna(subset=required_columns)
        print(f"After removing NaN values: {len(df_clean)} rows")
        
        if len(df_clean) == 0:
            print("No valid data found after cleaning!")
            return
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Define feature information
    features = [
        {
            'column': 'detected_angle',
            'title': 'Impact of Image Angle on OCR Accuracy',
            'xlabel': 'Detected Angle (degrees)',
            'ylabel': 'Average Address OCR Accuracy',
            'filename': 'angle_vs_accuracy.png'
        },
        {
            'column': 'detected_blur',
            'title': 'Impact of Image Blur on OCR Accuracy',
            'xlabel': 'Detected Blur Score',
            'ylabel': 'Average Address OCR Accuracy', 
            'filename': 'blur_vs_accuracy.png'
        },
        {
            'column': 'detected_brightness',
            'title': 'Impact of Image Brightness on OCR Accuracy',
            'xlabel': 'Detected Brightness',
            'ylabel': 'Average Address OCR Accuracy',
            'filename': 'brightness_vs_accuracy.png'
        },
        {
            'column': 'detected_reflection',
            'title': 'Impact of Reflection on OCR Accuracy',
            'xlabel': 'Reflection Status',
            'ylabel': 'Average Address OCR Accuracy',
            'filename': 'reflection_vs_accuracy.png'
        }
    ]
    
    for feature in features:
        # Create a new figure for each graph
        plt.figure(figsize=(10, 8))
        
        if feature['column'] == 'detected_reflection':
            # For reflection (boolean), create a bar chart showing average accuracy
            reflection_stats = df_clean.groupby('detected_reflection')['address_ocr_v8_accuracy'].agg(['mean', 'count', 'std']).reset_index()
            
            # Create labels
            labels = ['No Reflection', 'Has Reflection']
            means = reflection_stats['mean'].values
            counts = reflection_stats['count'].values
            stds = reflection_stats['std'].values
            
            # Create bar chart
            bars = plt.bar(labels, means, color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
            
            # Add error bars if we have standard deviations
            if not np.isnan(stds).any():
                plt.errorbar(labels, means, yerr=stds, fmt='none', color='black', capsize=5)
            
            # Add value labels on bars
            for i, (bar, mean_val, count) in enumerate(zip(bars, means, counts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean_val:.3f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
            
            # Calculate correlation or effect size
            reflection_effect = means[1] - means[0] if len(means) > 1 else 0
            plt.text(0.02, 0.98, f'Effect: {reflection_effect:.3f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
                    verticalalignment='top')
            
        else:
            # For continuous variables, create binned line graphs
            x_data = df_clean[feature['column']]
            y_data = df_clean['address_ocr_v8_accuracy']
            
            # Create bins for the x-axis data
            n_bins = 20
            bins = np.linspace(x_data.min(), x_data.max(), n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Calculate average accuracy for each bin
            bin_means = []
            bin_counts = []
            bin_stds = []
            
            for i in range(len(bins) - 1):
                mask = (x_data >= bins[i]) & (x_data < bins[i + 1])
                if i == len(bins) - 2:  # Include the last point in the last bin
                    mask = (x_data >= bins[i]) & (x_data <= bins[i + 1])
                
                bin_data = y_data[mask]
                if len(bin_data) > 0:
                    bin_means.append(bin_data.mean())
                    bin_counts.append(len(bin_data))
                    bin_stds.append(bin_data.std())
                else:
                    bin_means.append(np.nan)
                    bin_counts.append(0)
                    bin_stds.append(np.nan)
            
            # Remove bins with no data
            valid_indices = ~np.isnan(bin_means)
            bin_centers = bin_centers[valid_indices]
            bin_means = np.array(bin_means)[valid_indices]
            bin_counts = np.array(bin_counts)[valid_indices]
            bin_stds = np.array(bin_stds)[valid_indices]
            
            # Create line plot
            plt.plot(bin_centers, bin_means, 'b-', linewidth=2, marker='o', markersize=6, label='Average Accuracy')
            
            # Add confidence intervals (standard error)
            if len(bin_stds) > 0 and not np.isnan(bin_stds).all():
                standard_errors = bin_stds / np.sqrt(bin_counts)
                plt.fill_between(bin_centers, 
                               bin_means - standard_errors, 
                               bin_means + standard_errors, 
                               alpha=0.3, color='blue', label='Standard Error')
            
            # Calculate and display correlation
            try:
                correlation = np.corrcoef(x_data, y_data)[0, 1]
                plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                        transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
                        verticalalignment='top')
            except Exception as e:
                print(f"Could not calculate correlation for {feature['column']}: {e}")
            
            plt.legend()
        
        plt.title(feature['title'], fontsize=14, fontweight='bold', pad=20)
        plt.xlabel(feature['xlabel'], fontsize=12)
        plt.ylabel(feature['ylabel'], fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save the individual graph
        output_path = os.path.join(output_dir, feature['filename'])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close()  # Close the figure to free memory
    
    # Show comprehensive summary statistics
    print("\n" + "="*70)
    print("DETAILED IMPACT ANALYSIS")
    print("="*70)
    
    for feature in features:
        print(f"\n📊 {feature['title']}:")
        print("-" * 50)
        
        if feature['column'] == 'detected_reflection':
            try:
                reflection_stats = df_clean.groupby('detected_reflection')['address_ocr_v8_accuracy'].agg(['count', 'mean', 'std', 'min', 'max'])
                print(reflection_stats)
                
                # Calculate effect size
                no_reflection = df_clean[df_clean['detected_reflection'] == False]['address_ocr_v8_accuracy']
                has_reflection = df_clean[df_clean['detected_reflection'] == True]['address_ocr_v8_accuracy']
                
                if len(no_reflection) > 0 and len(has_reflection) > 0:
                    effect = has_reflection.mean() - no_reflection.mean()
                    print(f"\n💡 IMPACT: Reflection {'decreases' if effect < 0 else 'increases'} accuracy by {abs(effect):.3f}")
                    print(f"   📈 No Reflection Avg: {no_reflection.mean():.3f}")
                    print(f"   📉 Has Reflection Avg: {has_reflection.mean():.3f}")
                
            except Exception as e:
                print(f"Error calculating reflection stats: {e}")
        else:
            try:
                x_data = df_clean[feature['column']]
                y_data = df_clean['address_ocr_v8_accuracy']
                
                correlation = np.corrcoef(x_data, y_data)[0, 1]
                
                print(f"Correlation coefficient: {correlation:.4f}")
                print(f"Feature range: {x_data.min():.3f} to {x_data.max():.3f}")
                print(f"Accuracy range: {y_data.min():.3f} to {y_data.max():.3f}")
                
                # Interpretation of correlation
                if abs(correlation) > 0.7:
                    strength = "STRONG"
                elif abs(correlation) > 0.5:
                    strength = "MODERATE"
                elif abs(correlation) > 0.3:
                    strength = "WEAK"
                else:
                    strength = "VERY WEAK"
                
                direction = "POSITIVE" if correlation > 0 else "NEGATIVE"
                
                print(f"\n💡 IMPACT: {strength} {direction} correlation")
                if correlation > 0:
                    print(f"   📈 Higher {feature['xlabel'].lower()} → Higher accuracy")
                else:
                    print(f"   📉 Higher {feature['xlabel'].lower()} → Lower accuracy")
                
            except Exception as e:
                print(f"Error calculating stats for {feature['column']}: {e}")
    
    print(f"\n✅ All graphs saved in '{output_dir}' directory")

def main():
    # Use your CSV file path
    csv_file_path = "nid_infos_compared_03-08-25_15-34.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        print("Please update the file path in the script.")
        return
    
    create_feature_analysis_graphs(csv_file_path)

if __name__ == "__main__":
    main()