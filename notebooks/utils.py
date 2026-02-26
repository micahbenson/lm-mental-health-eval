import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

def clean_df(df): 
    # Normalize nested columns
    df['prompt_id'] = df['doc'].apply(lambda x: x['id'])
    df['prompt_text'] = df['doc'].apply(lambda x: x['prompt_text'])
    df['disorder'] = df['doc'].apply(lambda x: x['tags']['disorder'])
    df['symptom'] = df['doc'].apply(lambda x: x['tags']['symptom'])
    df['severity'] = df['doc'].apply(lambda x: x['tags']['severity'])
    df['round'] = df['doc'].apply(lambda x: x['tags']['rephrase_source'])
    df['response'] = df['filtered_resps'].apply(lambda x: x[0] if x else None)

    if 'jailbreak_category' in df['doc'].iloc[0].get('tags', {}): 
        df['jailbreak_category'] = df['doc'].apply(lambda x: x['tags']['jailbreak_category'])
    #Cutting out the original BDI statements because they're off distribution
    df = df[df['round']!='original_text']
    return df 


def plot_grouped_bar(
    model_paths: Dict[str, List[str]],
    metric: str,
    title: str,
    ylabel: str,
    figsize: Tuple[int, int] = (10, 5),
    color_palette: str = 'colorblind',
    filters: Optional[dict] = None,
    save_path: Optional[str] = None
):
    """
    Create a grouped bar chart comparing a metric across multiple models.
    
    Parameters
    ----------
    model_paths : Dict[str, List[str]]
        Dictionary mapping model display names to lists of JSONL file paths.
        Example: {'Olmo-3-7B': [bdi_path, bai_path], 'Llama-3.1-8B': [bdi_path, bai_path]}
    metric : str, default 'hotline_rate'
        The metric column to plot (e.g., 'hotline_rate', 'referral_rate', 'awareness_rate')
    title : str, optional
        Plot title. If None, generates a default title.
    ylabel : str, optional
        Y-axis label. If None, uses the metric name formatted nicely.
    figsize : Tuple[int, int], default (10, 5)
        Figure size in inches
    color_palette : str, default 'colorblind'
        Seaborn color palette name
    save_path : str, optional
        If provided, saves the figure to this path
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    # Load and aggregate data for each model
    model_aggs = {}
    
    for model_name, paths in model_paths.items():
        # Load and concatenate all files for this model
        dfs = [clean_df(pd.read_json(path, lines=True)) for path in paths]

        #Allow filtering!
        if filters is not None:
            for col, value in filters.items():
                dfs = [df[df[col]==value] for df in dfs]

        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Calculate aggregated statistics
        agg = combined_df.groupby('severity')[metric].agg(['mean', 'sem'])
        model_aggs[model_name] = agg
    
    
    # Setup plot
    x = np.arange(4)
    severity_labels = ['none', 'mild', 'moderate', 'severe']
    n_models = len(model_aggs)
    width = 0.8 / n_models  # Dynamic width based on number of models
    colors = sns.color_palette(color_palette, n_models)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bar positions (centered around x)
    offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width
    
    # Create bars for each model
    for idx, (model_name, agg_data) in enumerate(model_aggs.items()):
        ax.bar(
            x + offsets[idx],
            agg_data['mean'],
            width,
            yerr=agg_data['sem'],
            capsize=4,
            label=model_name,
            alpha=0.8,
            color=colors[idx]
        )
    
    # Format plot
    ax.set_ylim(0, 1)
    ax.set_xlabel('Prompt Statement Severity', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=17)

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(severity_labels, fontsize=12)
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_before_after(
    model_paths: Dict[str, Dict[str, List[str]]],
    metric: str,
    title: str,
    ylabel: str,
    figsize: Tuple[int, int] = (10, 5),
    color_palette: str = 'colorblind',
    before_filters: Optional[dict] = None,
    after_filters: Optional[dict] = None,
    save_path: Optional[str] = None,
):
    """
    Create a bar chart comparing before/after values for a metric across multiple models.
    Bars are superimposed with distinct styling to show change.
    
    Parameters
    ----------
    model_paths : Dict[str, Dict[str, List[str]]]
        Dictionary mapping model names to before/after JSONL paths.
        Example: {
            'Olmo-3-7B': {
                'before': [bdi_path, bai_path],
                'after': [bdi_path_after, bai_path_after]
            }
        }
    metric : str
        The metric column to plot (e.g., 'hotline_rate', 'referral_rate', 'awareness_rate')
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : Tuple[int, int], default (10, 5)
        Figure size in inches
    color_palette : str, default 'colorblind'
        Seaborn color palette name
    filters : dict, optional
        Column filters to apply to data
    save_path : str, optional
        If provided, saves the figure to this path
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    # Load and aggregate data for each model and condition
    model_aggs = {}
    
    for model_name, conditions in model_paths.items():
        model_aggs[model_name] = {}
        
        for condition in ['before', 'after']:
            # Load and concatenate all files for this model/condition
            dfs = [clean_df(pd.read_json(path, lines=True)) 
                   for path in conditions[condition]]
            
            if condition == 'before': 
                # Apply filters
                if before_filters is not None:
                    for col, value in before_filters.items():
                        dfs = [df[df[col] == value] for df in dfs]
            
            if condition == 'after': 
                # Apply filters
                if after_filters is not None:
                    for col, value in after_filters.items():
                        dfs = [df[df[col] == value] for df in dfs]
            
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Calculate aggregated statistics
            agg = combined_df.groupby('severity')[metric].agg(['mean', 'sem'])
            model_aggs[model_name][condition] = agg
    
    # Setup plot
    x = np.arange(4)
    severity_labels = ['none', 'mild', 'moderate', 'severe']
    n_models = len(model_aggs)
    width = 0.8 / n_models  # Dynamic width based on number of models
    colors = sns.color_palette(color_palette, n_models)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bar positions (centered around x)
    offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width
    
    before_alpha = 0.4
    after_alpha = 0.9

    # Create bars for each model
    for idx, (model_name, agg_data) in enumerate(model_aggs.items()):
        base_color = colors[idx]
        x_pos = x + offsets[idx]
        
        # Plot "before" bars first (lighter, in back)
        ax.bar(
            x_pos,
            agg_data['before']['mean'],
            width,
            yerr=agg_data['before']['sem'],
            capsize=4,
            label=f'{model_name} (Before)',
            alpha=before_alpha,
            color=base_color,
            edgecolor=base_color,
            linewidth=1.5
        )
        
        # Plot "after" bars on top (darker, in front)
        ax.bar(
            x_pos,
            agg_data['after']['mean'],
            width,
            yerr=agg_data['after']['sem'],
            capsize=4,
            label=f'{model_name} (After)',
            alpha=after_alpha,
            color=base_color,
            edgecolor='black',
            linewidth=1.5
        )
    
    # Format plot
    ax.set_ylim(0, 1)
    ax.set_xlabel('Prompt Statement Severity', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=17)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(severity_labels, fontsize=12)
    
    # Organize legend: group by model
    handles, labels = ax.get_legend_handles_labels()
    # Reorder to group before/after pairs together
    legend_order = []
    for i in range(0, len(handles), 2):
        legend_order.extend([i, i+1])
    ax.legend([handles[i] for i in legend_order], 
              [labels[i] for i in legend_order],
              loc='best', fontsize=10)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def heatmap(
    file : str, 
    metric : str,
    title : str,
    figsize : Tuple[int, int] = (5, 8),
    filters : Optional[dict] = None,
): 
    df = pd.read_json(file, lines=True) 
    df = clean_df(df)

    #Allow filtering!
    if filters is not None:
        for col, value in filters.items():
            df = df[df[col]==value]

    # Pivot the data: symptom as rows, severity as columns
    heatmap_data = df.groupby(['symptom', 'severity'])[metric].mean().unstack()

    # Create heatmap
    plt.figure(figsize=figsize)  # Adjust size as needed
    sns.heatmap(heatmap_data, 
                annot=True,           # Show values in cells
                fmt='.2f',            # Format to 3 decimal places
                cmap='YlOrRd',        # Color scheme
                vmin=0, 
                vmax=1,
                cbar_kws={'label': f'{metric}'},
                linewidths=0.5)       # Grid lines

    plt.title(title)
    plt.xlabel('Severity')
    plt.ylabel('Symptom')
    plt.tight_layout()
    plt.show()


def heatmap_diff(file, og_file, metric): 
    df = pd.read_json(file, lines=True) 
    df = clean_df(df)

    og_df = pd.read_json(og_file, lines=True) 
    og_df = clean_df(df)
    
    for jailbreak in df['jailbreak_category'].unique(): 
        category_df = df[df['jailbreak_category']==jailbreak]        
        # Pivot the data: symptom as rows, severity as columns
        heatmap_data = category_df.groupby(['symptom', 'severity'])[metric].mean().unstack()

        # Create heatmap
        plt.figure(figsize=(5, 8))  # Adjust size as needed
        sns.heatmap(heatmap_data, 
                    annot=True,           # Show values in cells
                    fmt='.2f',            # Format to 3 decimal places
                    cmap='YlOrRd',        # Color scheme
                    vmin=0, 
                    vmax=1,
                    cbar_kws={'label': f'{metric}'},
                    linewidths=0.5)       # Grid lines

        plt.title(title)
        plt.xlabel('Severity')
        plt.ylabel('Symptom')
        plt.tight_layout()
        plt.show()