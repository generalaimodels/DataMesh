"""
SOTA Retrieval Visualization: UMAP + Plotly Interactive Embedding Space

Proof-of-concept visualization demonstrating hybrid retrieval effectiveness:
    - UMAP projection of high-dimensional embeddings to 2D/3D
    - Interactive Plotly scatter with document clusters
    - Query vector positioning showing retrieval relevance
    - Animated query→result connections

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    VISUALIZATION PIPELINE                    │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  [Dense Embeddings] → [UMAP] → [2D/3D Projection]           │
    │                                       ↓                      │
    │  [Query Embedding]  → [UMAP Transform] → Query Point        │
    │                                       ↓                      │
    │                          [Plotly Interactive Chart]          │
    │                                       ↓                      │
    │                     - Document clusters (color-coded)        │
    │                     - Query position (star marker)           │
    │                     - Retrieved docs (connected/highlighted) │
    │                     - Hover tooltips with doc content        │
    └─────────────────────────────────────────────────────────────┘

Features:
    - Semantic clustering: Related docs cluster together
    - Query proximity: Query appears near relevant documents
    - Interactive exploration: Zoom, pan, hover for details
    - Multi-query overlay: Compare multiple queries

References:
    - UMAP: McInnes et al., arXiv 2018
    - Embedding Visualization: Nolet et al., EMNLP 2020
"""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

try:
    import umap
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("   Install with: pip install umap-learn plotly")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    # UMAP parameters
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 2  # 2D or 3D
    metric: str = "cosine"
    
    # Plotly styling
    width: int = 1200
    height: int = 800
    marker_size: int = 8
    query_marker_size: int = 20
    opacity: float = 0.7
    
    # Colors
    color_scheme: str = "Viridis"
    query_color: str = "#FF4444"
    retrieved_color: str = "#00FF88"
    connection_color: str = "rgba(255, 100, 100, 0.4)"
    
    # Export
    output_dir: str = "./visualizations"
    save_html: bool = True


# =============================================================================
# UMAP PROJECTOR
# =============================================================================
class EmbeddingProjector:
    """
    UMAP-based dimensionality reduction for embedding visualization.
    
    Maps high-dimensional embeddings (768D, 1024D) to 2D/3D space
    while preserving local neighborhood structure.
    """
    
    __slots__ = ('_config', '_reducer', '_fitted', '_corpus_embeddings')
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self._config = config or VisualizationConfig()
        self._reducer: Optional[umap.UMAP] = None
        self._fitted = False
        self._corpus_embeddings: Optional[np.ndarray] = None
    
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit UMAP on corpus embeddings and return projections.
        
        Args:
            embeddings: (N, D) array of document embeddings
            
        Returns:
            (N, 2|3) projected coordinates
        """
        print(f"   🔄 Fitting UMAP (n={len(embeddings)}, dim={embeddings.shape[1]})...")
        start = time.perf_counter()
        
        self._reducer = umap.UMAP(
            n_neighbors=self._config.n_neighbors,
            min_dist=self._config.min_dist,
            n_components=self._config.n_components,
            metric=self._config.metric,
            random_state=42,
            n_jobs=-1,
        )
        
        projections = self._reducer.fit_transform(embeddings)
        self._corpus_embeddings = embeddings
        self._fitted = True
        
        elapsed = time.perf_counter() - start
        print(f"   ✅ UMAP completed in {elapsed:.2f}s")
        
        return projections
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project new embeddings (e.g., queries) using fitted UMAP.
        
        Args:
            embeddings: (M, D) array of new embeddings
            
        Returns:
            (M, 2|3) projected coordinates
        """
        if not self._fitted:
            raise RuntimeError("Projector not fitted. Call fit() first.")
        
        return self._reducer.transform(embeddings)


# =============================================================================
# VISUALIZATION BUILDER
# =============================================================================
class RetrievalVisualizer:
    """
    Interactive Plotly visualization for retrieval results.
    
    Creates publication-quality visualizations showing:
        - Document embeddings as clustered points
        - Query position in embedding space
        - Retrieved documents highlighted with connections
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self._config = config or VisualizationConfig()
        self._projector = EmbeddingProjector(config)
    
    def create_corpus_visualization(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        documents: Optional[List[str]] = None,
    ) -> Tuple[go.Figure, np.ndarray]:
        """
        Create interactive corpus embedding visualization.
        
        Args:
            embeddings: (N, D) document embeddings
            labels: Document labels for hover
            categories: Category labels for coloring
            documents: Document text snippets for tooltips
            
        Returns:
            (Plotly Figure, projected coordinates)
        """
        # Project to 2D
        projections = self._projector.fit(embeddings)
        
        # Prepare data
        n_docs = len(embeddings)
        if labels is None:
            labels = [f"Doc {i}" for i in range(n_docs)]
        if categories is None:
            categories = ["corpus"] * n_docs
        if documents is None:
            documents = ["" for _ in range(n_docs)]
        
        # Truncate documents for hover
        hover_texts = [
            f"<b>{label}</b><br>{doc[:100]}..." if len(doc) > 100 else f"<b>{label}</b><br>{doc}"
            for label, doc in zip(labels, documents)
        ]
        
        # Create figure
        if self._config.n_components == 2:
            fig = self._create_2d_scatter(
                projections, labels, categories, hover_texts
            )
        else:
            fig = self._create_3d_scatter(
                projections, labels, categories, hover_texts
            )
        
        return fig, projections
    
    def _create_2d_scatter(
        self,
        projections: np.ndarray,
        labels: List[str],
        categories: List[str],
        hover_texts: List[str],
    ) -> go.Figure:
        """Create 2D scatter visualization."""
        # Get unique categories and assign colors
        unique_cats = list(set(categories))
        color_map = {cat: i for i, cat in enumerate(unique_cats)}
        colors = [color_map[cat] for cat in categories]
        
        fig = go.Figure()
        
        # Add document points
        fig.add_trace(go.Scatter(
            x=projections[:, 0],
            y=projections[:, 1],
            mode='markers',
            marker=dict(
                size=self._config.marker_size,
                color=colors,
                colorscale=self._config.color_scheme,
                opacity=self._config.opacity,
                line=dict(width=0.5, color='white'),
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Documents',
        ))
        
        # Styling
        fig.update_layout(
            title=dict(
                text='<b>SOTA Retrieval: Embedding Space Visualization</b>',
                font=dict(size=20, color='#333'),
                x=0.5,
            ),
            xaxis=dict(
                title='UMAP Dimension 1',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
            ),
            yaxis=dict(
                title='UMAP Dimension 2',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
            ),
            width=self._config.width,
            height=self._config.height,
            template='plotly_white',
            hovermode='closest',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
            ),
        )
        
        return fig
    
    def _create_3d_scatter(
        self,
        projections: np.ndarray,
        labels: List[str],
        categories: List[str],
        hover_texts: List[str],
    ) -> go.Figure:
        """Create 3D scatter visualization."""
        unique_cats = list(set(categories))
        color_map = {cat: i for i, cat in enumerate(unique_cats)}
        colors = [color_map[cat] for cat in categories]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=projections[:, 0],
            y=projections[:, 1],
            z=projections[:, 2],
            mode='markers',
            marker=dict(
                size=self._config.marker_size,
                color=colors,
                colorscale=self._config.color_scheme,
                opacity=self._config.opacity,
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Documents',
        )])
        
        fig.update_layout(
            title='<b>SOTA Retrieval: 3D Embedding Space</b>',
            width=self._config.width,
            height=self._config.height,
            template='plotly_white',
        )
        
        return fig
    
    def add_query_and_results(
        self,
        fig: go.Figure,
        corpus_projections: np.ndarray,
        query_embedding: np.ndarray,
        retrieved_indices: List[int],
        query_text: str = "Query",
        scores: Optional[List[float]] = None,
    ) -> go.Figure:
        """
        Add query point and highlight retrieved documents.
        
        Args:
            fig: Existing corpus visualization
            corpus_projections: (N, 2|3) corpus projections
            query_embedding: (D,) query vector
            retrieved_indices: Indices of retrieved documents
            query_text: Query text for label
            scores: Optional retrieval scores
            
        Returns:
            Updated figure with query and connections
        """
        # Project query
        query_proj = self._projector.transform(query_embedding.reshape(1, -1))[0]
        
        is_3d = self._config.n_components == 3
        
        # Add connections to retrieved docs
        for i, idx in enumerate(retrieved_indices[:10]):  # Top 10
            doc_proj = corpus_projections[idx]
            score = scores[i] if scores else 1.0
            
            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=[query_proj[0], doc_proj[0]],
                    y=[query_proj[1], doc_proj[1]],
                    z=[query_proj[2], doc_proj[2]],
                    mode='lines',
                    line=dict(color=self._config.connection_color, width=2 * score),
                    showlegend=False,
                    hoverinfo='skip',
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[query_proj[0], doc_proj[0]],
                    y=[query_proj[1], doc_proj[1]],
                    mode='lines',
                    line=dict(color=self._config.connection_color, width=2 + 3 * score),
                    showlegend=False,
                    hoverinfo='skip',
                ))
        
        # Highlight retrieved documents
        retrieved_projs = corpus_projections[retrieved_indices]
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=retrieved_projs[:, 0],
                y=retrieved_projs[:, 1],
                z=retrieved_projs[:, 2],
                mode='markers',
                marker=dict(
                    size=self._config.marker_size + 4,
                    color=self._config.retrieved_color,
                    symbol='diamond',
                    line=dict(width=2, color='#00AA66'),
                ),
                name='Retrieved',
            ))
        else:
            fig.add_trace(go.Scatter(
                x=retrieved_projs[:, 0],
                y=retrieved_projs[:, 1],
                mode='markers',
                marker=dict(
                    size=self._config.marker_size + 6,
                    color=self._config.retrieved_color,
                    symbol='diamond',
                    line=dict(width=2, color='#00AA66'),
                ),
                name='Retrieved',
            ))
        
        # Add query point (star marker)
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=[query_proj[0]],
                y=[query_proj[1]],
                z=[query_proj[2]],
                mode='markers+text',
                marker=dict(
                    size=self._config.query_marker_size,
                    color=self._config.query_color,
                    symbol='diamond',
                    line=dict(width=2, color='#AA0000'),
                ),
                text=[f'Q: {query_text[:30]}...'],
                textposition='top center',
                name='Query',
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[query_proj[0]],
                y=[query_proj[1]],
                mode='markers+text',
                marker=dict(
                    size=self._config.query_marker_size,
                    color=self._config.query_color,
                    symbol='star',
                    line=dict(width=2, color='#AA0000'),
                ),
                text=[f'🔍 {query_text[:40]}'],
                textposition='top center',
                textfont=dict(size=12, color='#CC0000'),
                name='Query',
            ))
        
        return fig
    
    def create_multi_query_comparison(
        self,
        corpus_projections: np.ndarray,
        queries: List[Tuple[str, np.ndarray, List[int]]],
        documents: Optional[List[str]] = None,
    ) -> go.Figure:
        """
        Create visualization comparing multiple queries.
        
        Args:
            corpus_projections: (N, 2) document projections
            queries: List of (query_text, embedding, retrieved_indices)
            documents: Document texts for hover
            
        Returns:
            Multi-query comparison figure
        """
        colors = px.colors.qualitative.Set1
        
        fig = go.Figure()
        
        # Add corpus points
        hover_texts = [f"Doc {i}" for i in range(len(corpus_projections))]
        if documents:
            hover_texts = [doc[:100] for doc in documents]
        
        fig.add_trace(go.Scatter(
            x=corpus_projections[:, 0],
            y=corpus_projections[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='rgba(100, 100, 100, 0.3)',
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Corpus',
        ))
        
        # Add each query with its results
        for i, (q_text, q_emb, retrieved) in enumerate(queries):
            color = colors[i % len(colors)]
            q_proj = self._projector.transform(q_emb.reshape(1, -1))[0]
            
            # Retrieved points
            ret_proj = corpus_projections[retrieved[:5]]
            fig.add_trace(go.Scatter(
                x=ret_proj[:, 0],
                y=ret_proj[:, 1],
                mode='markers',
                marker=dict(size=12, color=color, symbol='circle'),
                name=f'Q{i+1} Results',
            ))
            
            # Query point
            fig.add_trace(go.Scatter(
                x=[q_proj[0]],
                y=[q_proj[1]],
                mode='markers+text',
                marker=dict(size=18, color=color, symbol='star'),
                text=[f'Q{i+1}'],
                textposition='top center',
                name=f'Query: {q_text[:20]}...',
            ))
            
            # Connections
            for idx in retrieved[:3]:
                doc_proj = corpus_projections[idx]
                fig.add_trace(go.Scatter(
                    x=[q_proj[0], doc_proj[0]],
                    y=[q_proj[1], doc_proj[1]],
                    mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip',
                ))
        
        fig.update_layout(
            title='<b>Multi-Query Comparison: Retrieval Patterns</b>',
            width=self._config.width,
            height=self._config.height,
            template='plotly_white',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: str) -> str:
        """Save figure to HTML file."""
        os.makedirs(self._config.output_dir, exist_ok=True)
        filepath = os.path.join(self._config.output_dir, f"{filename}.html")
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"   💾 Saved: {filepath}")
        return filepath


# =============================================================================
# DEMO PIPELINE
# =============================================================================
async def run_visualization_demo(max_samples: int = 500) -> None:
    """
    Run complete visualization demo.
    
    1. Load documents and embeddings
    2. Project to 2D with UMAP
    3. Create interactive visualization
    4. Demonstrate query positioning
    """
    from sentence_transformers import SentenceTransformer
    
    print("\n🎨 SOTA Retrieval Visualization Demo")
    print("=" * 60)
    
    # Load model
    print("   📦 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create sample corpus
    print("   📄 Creating sample corpus...")
    corpus_texts = [
        # Tech cluster
        "Machine learning is a subset of artificial intelligence",
        "Deep neural networks revolutionized computer vision",
        "Natural language processing enables chatbots",
        "Reinforcement learning powers game-playing AI",
        "Transformers architecture changed NLP forever",
        "Computer vision detects objects in images",
        "Speech recognition converts audio to text",
        "Recommendation systems suggest relevant content",
        
        # Science cluster
        "Quantum mechanics describes particle behavior",
        "Black holes are regions of extreme gravity",
        "DNA contains genetic information",
        "Evolution explains species diversity",
        "Climate change affects global temperatures",
        "Renewable energy reduces carbon emissions",
        "Vaccine development saves millions of lives",
        "Space exploration expands human knowledge",
        
        # Business cluster  
        "Stock markets fluctuate based on sentiment",
        "Marketing strategies drive customer engagement",
        "Supply chain optimization reduces costs",
        "Leadership skills inspire team performance",
        "Financial planning ensures long-term stability",
        "Customer service builds brand loyalty",
        "Product development meets market needs",
        "Strategic partnerships accelerate growth",
    ]
    
    categories = (
        ["Technology"] * 8 + 
        ["Science"] * 8 + 
        ["Business"] * 8
    )
    
    # Generate embeddings
    print(f"   🔄 Generating embeddings for {len(corpus_texts)} documents...")
    embeddings = model.encode(corpus_texts, show_progress_bar=False)
    embeddings = np.array(embeddings)
    
    # Create visualizer
    config = VisualizationConfig(
        n_neighbors=5,
        min_dist=0.3,
    )
    visualizer = RetrievalVisualizer(config)
    
    # Create corpus visualization
    print("\n📊 Creating Embedding Space Visualization...")
    fig, projections = visualizer.create_corpus_visualization(
        embeddings,
        labels=[f"Doc {i}" for i in range(len(corpus_texts))],
        categories=categories,
        documents=corpus_texts,
    )
    
    # Save initial visualization
    corpus_path = visualizer.save_figure(fig, "corpus_embeddings")
    
    # Demo query
    print("\n🔍 Adding Query Visualization...")
    query = "How does artificial intelligence learn from data?"
    query_emb = model.encode([query])[0]
    
    # Find most similar documents (simple cosine similarity)
    similarities = np.dot(embeddings, query_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    retrieved_indices = np.argsort(similarities)[::-1][:5].tolist()
    scores = similarities[retrieved_indices].tolist()
    
    # Add query to visualization
    fig_with_query = visualizer.add_query_and_results(
        fig,
        projections,
        query_emb,
        retrieved_indices,
        query_text=query,
        scores=scores,
    )
    
    query_path = visualizer.save_figure(fig_with_query, "query_retrieval")
    
    # Multi-query comparison
    print("\n📈 Creating Multi-Query Comparison...")
    queries = [
        ("AI and machine learning", model.encode(["AI and machine learning"])[0]),
        ("Climate and environment", model.encode(["Climate and environment"])[0]),
        ("Business strategy", model.encode(["Business strategy"])[0]),
    ]
    
    multi_query_data = []
    for q_text, q_emb in queries:
        sims = np.dot(embeddings, q_emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
        )
        ret_idx = np.argsort(sims)[::-1][:5].tolist()
        multi_query_data.append((q_text, q_emb, ret_idx))
    
    multi_fig = visualizer.create_multi_query_comparison(
        projections, multi_query_data, corpus_texts
    )
    multi_path = visualizer.save_figure(multi_fig, "multi_query_comparison")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Visualization Complete!")
    print("=" * 60)
    print(f"\n📁 Generated Files:")
    print(f"   1. {corpus_path}")
    print(f"   2. {query_path}")
    print(f"   3. {multi_path}")
    print("\n🌐 Open any .html file in your browser to explore interactively!")
    
    # Show in browser
    import webbrowser
    webbrowser.open(f'file://{os.path.abspath(query_path)}')


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SOTA Retrieval Visualization with UMAP + Plotly",
    )
    parser.add_argument(
        "--max-samples", type=int, default=500,
        help="Maximum documents to visualize"
    )
    parser.add_argument(
        "--3d", action="store_true", dest="use_3d",
        help="Use 3D visualization"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./visualizations",
        help="Output directory for HTML files"
    )
    
    args = parser.parse_args()
    asyncio.run(run_visualization_demo(args.max_samples))


if __name__ == "__main__":
    main()
