import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import os
from pathlib import Path

pio.templates.default = "plotly_white"

def load_rotation_data():
    logs_dir = Path(__file__).parent / "logs"

    front_df = pd.read_csv(logs_dir / "front_rotations.csv")
    back_df = pd.read_csv(logs_dir / "back_rotations.csv")

    front_df['side'] = 'Front'
    back_df['side'] = 'Back'

    combined_df = pd.concat([front_df, back_df], ignore_index=True)

    return front_df, back_df, combined_df

def create_angle_distribution_plots(front_df, back_df, combined_df):
    fig1 = px.histogram(combined_df, x="applied_angle", color="side",
                       marginal="box",
                       title="Distribution of Applied Rotation Angles",
                       labels={"applied_angle": "Rotation Angle (degrees)", "count": "Number of Images"},
                       color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                       opacity=0.7)

    fig1.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, color='#2C3E50'),
        showlegend=True,
        legend_title_text='Image Side'
    )

    fig2 = px.violin(combined_df, y="applied_angle", x="side", color="side",
                    title="Angle Distribution Comparison (Violin Plot)",
                    labels={"applied_angle": "Rotation Angle (degrees)", "side": "Image Side"},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                    box=True, points="all")

    fig2.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, color='#2C3E50')
    )

    return fig1, fig2

def create_statistics_summary(front_df, back_df, combined_df):
    front_stats = front_df['applied_angle'].describe()
    back_stats = back_df['applied_angle'].describe()

    stats_data = []
    for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        stats_data.extend([
            {'Statistic': stat, 'Value': front_stats[stat], 'Side': 'Front'},
            {'Statistic': stat, 'Value': back_stats[stat], 'Side': 'Back'}
        ])

    stats_df = pd.DataFrame(stats_data)

    fig = px.bar(stats_df, x='Statistic', y='Value', color='Side',
                title='Statistical Comparison: Front vs Back Rotation Angles',
                barmode='group',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'])

    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, color='#2C3E50'),
        xaxis_title="Statistical Measure",
        yaxis_title="Value (degrees)"
    )

    return fig

def create_angle_range_analysis(front_df, back_df):
    bins = [-15, -10, -5, 0, 5, 10, 15]
    labels = ['-15° to -10°', '-10° to -5°', '-5° to 0°', '0° to 5°', '5° to 10°', '10° to 15°']

    front_df['angle_range'] = pd.cut(front_df['applied_angle'], bins=bins, labels=labels)
    back_df['angle_range'] = pd.cut(back_df['applied_angle'], bins=bins, labels=labels)

    front_counts = front_df['angle_range'].value_counts().sort_index()
    back_counts = back_df['angle_range'].value_counts().sort_index()

    range_comparison = pd.DataFrame({
        'Angle Range': labels,
        'Front Images': front_counts,
        'Back Images': back_counts
    }).fillna(0)

    range_melted = range_comparison.melt(id_vars='Angle Range',
                                       var_name='Image Side',
                                       value_name='Count')

    fig = px.bar(range_melted, x='Angle Range', y='Count', color='Image Side',
                title='Rotation Angle Ranges: Front vs Back Images',
                barmode='group',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'])

    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, color='#2C3E50'),
        xaxis_title="Rotation Angle Range",
        yaxis_title="Number of Images"
    )

    return fig

def create_image_size_analysis(combined_df):
    def extract_dimensions(shape_str):
        dims = shape_str.strip('()').split(', ')
        return int(dims[0]), int(dims[1])

    combined_df[['height', 'width']] = combined_df['saved_shape'].apply(
        lambda x: pd.Series(extract_dimensions(x))
    )

    combined_df['area'] = combined_df['height'] * combined_df['width']

    fig1 = px.scatter(combined_df, x='applied_angle', y='area', color='side',
                     title='Rotation Angle vs Image Area',
                     labels={'applied_angle': 'Rotation Angle (degrees)',
                            'area': 'Image Area (pixels)',
                            'side': 'Image Side'},
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4'])

    fig1.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, color='#2C3E50')
    )

    fig2 = px.box(combined_df, x='side', y='area', color='side',
                 title='Image Area Distribution: Front vs Back',
                 labels={'area': 'Image Area (pixels)', 'side': 'Image Side'},
                 color_discrete_sequence=['#FF6B6B', '#4ECDC4'])

    fig2.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, color='#2C3E50')
    )

    return fig1, fig2

def create_comprehensive_dashboard(front_df, back_df, combined_df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Angle Distribution', 'Angle Ranges',
                       'Image Areas', 'Angle vs Area Correlation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    for side, color in [('Front', '#FF6B6B'), ('Back', '#4ECDC4')]:
        df_side = combined_df[combined_df['side'] == side]
        fig.add_trace(
            go.Histogram(x=df_side['applied_angle'], name=f'{side} Angles',
                        marker_color=color, opacity=0.7),
            row=1, col=1
        )

    bins = [-15, -10, -5, 0, 5, 10, 15]
    labels = ['-15° to -10°', '-10° to -5°', '-5° to 0°', '0° to 5°', '5° to 10°', '10° to 15°']

    for side, color in [('Front', '#FF6B6B'), ('Back', '#4ECDC4')]:
        df_side = combined_df[combined_df['side'] == side]
        df_side['angle_range'] = pd.cut(df_side['applied_angle'], bins=bins, labels=labels)
        range_counts = df_side['angle_range'].value_counts().sort_index()

        fig.add_trace(
            go.Bar(x=labels, y=range_counts, name=f'{side} Ranges',
                  marker_color=color, opacity=0.7, showlegend=False),
            row=1, col=2
        )

    for side, color in [('Front', '#FF6B6B'), ('Back', '#4ECDC4')]:
        df_side = combined_df[combined_df['side'] == side].copy()
        dims = df_side['saved_shape'].apply(lambda x: x.strip('()').split(', ')).apply(lambda x: (int(x[0]), int(x[1])))
        df_side['height'] = dims.apply(lambda x: x[0])
        df_side['width'] = dims.apply(lambda x: x[1])
        df_side['area'] = df_side['height'] * df_side['width']

        fig.add_trace(
            go.Box(y=df_side['area'], name=f'{side} Areas',
                  marker_color=color, showlegend=False),
            row=2, col=1
        )

    for side, color in [('Front', '#FF6B6B'), ('Back', '#4ECDC4')]:
        df_side = combined_df[combined_df['side'] == side].copy()
        dims = df_side['saved_shape'].apply(lambda x: x.strip('()').split(', ')).apply(lambda x: (int(x[0]), int(x[1])))
        df_side['height'] = dims.apply(lambda x: x[0])
        df_side['width'] = dims.apply(lambda x: x[1])
        df_side['area'] = df_side['height'] * df_side['width']

        fig.add_trace(
            go.Scatter(x=df_side['applied_angle'], y=df_side['area'],
                      mode='markers', name=f'{side} Correlation',
                      marker=dict(color=color, size=6, opacity=0.6)),
            row=2, col=2
        )

    fig.update_layout(
        title_text="Comprehensive Rotation Analysis Dashboard",
        title_font=dict(size=24, color='#2C3E50'),
        font=dict(size=12),
        showlegend=True,
        height=800
    )

    fig.update_xaxes(title_text="Rotation Angle (degrees)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Angle Range", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Image Area (pixels)", row=2, col=1)
    fig.update_xaxes(title_text="Rotation Angle (degrees)", row=2, col=2)
    fig.update_yaxes(title_text="Image Area (pixels)", row=2, col=2)

    return fig

def save_plots_to_files():
    print("🔄 Loading rotation data...")
    front_df, back_df, combined_df = load_rotation_data()

    print(f"✅ Loaded {len(front_df)} front images and {len(back_df)} back images")

    visuals_dir = Path(__file__).parent / "visuals"

    print("📊 Creating angle distribution plots...")
    hist_fig, violin_fig = create_angle_distribution_plots(front_df, back_df, combined_df)
    hist_fig.write_image(visuals_dir / "angle_distribution_histogram.png", format='png', width=1000, height=600)
    violin_fig.write_image(visuals_dir / "angle_distribution_violin.png", format='png', width=1000, height=600)

    print("📈 Creating statistics summary...")
    stats_fig = create_statistics_summary(front_df, back_df, combined_df)
    stats_fig.write_image(visuals_dir / "angle_statistics_comparison.png", format='png', width=1000, height=600)

    print("🎯 Creating angle range analysis...")
    range_fig = create_angle_range_analysis(front_df, back_df)
    range_fig.write_image(visuals_dir / "angle_ranges_comparison.png", format='png', width=1000, height=600)

    print("📏 Creating image size analysis...")
    scatter_fig, box_fig = create_image_size_analysis(combined_df)
    scatter_fig.write_image(visuals_dir / "angle_vs_area_scatter.png", format='png', width=1000, height=600)
    box_fig.write_image(visuals_dir / "image_area_distribution.png", format='png', width=1000, height=600)

    print("🎨 Creating comprehensive dashboard...")
    dashboard_fig = create_comprehensive_dashboard(front_df, back_df, combined_df)
    dashboard_fig.write_image(visuals_dir / "comprehensive_dashboard.png", format='png', width=1200, height=800)

    print("\n✅ All visualizations saved to 'visuals/' directory!")
    print("📁 Generated files:")
    for file in sorted(visuals_dir.glob("*.png")):
        print(f"   • {file.name}")

    print("\n🔍 Key Insights:")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")

if __name__ == "__main__":
    save_plots_to_files()