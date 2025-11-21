"""
Taxonomy hierarchy management for hierarchical classification.

This module handles building and managing the taxonomic hierarchy from
the Catalogue of Life (COL) or database. It creates a tree structure
used for hierarchical YOLO model training and inference.

Corresponds to Methods section "Hierarchical Classification Architecture".
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import json

from src.config import config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TaxonomyHierarchy:
    """
    Build and manage taxonomic hierarchy for classification.

    This class creates a tree structure of taxa based on parent-child
    relationships, filters by time period, and aggregates specimen images
    at each taxonomic level.

    Example:
        >>> hierarchy = TaxonomyHierarchy(start_year=2020, end_year=2023)
        >>> taxa_tree = hierarchy.build()
        >>> hierarchy.print_tree(taxa_tree)
        Arthropoda (ID: RT, Total images: 5234)
            Insecta (ID: 234, Total images: 4521)
                Coleoptera (ID: 567, Total images: 1234)
                    ...
    """

    def __init__(
        self,
        data_manager=None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ):
        """
        Initialize taxonomy hierarchy builder.

        Args:
            data_manager: DataManager instance for database/file access
            start_year: Filter specimens from this year onwards (inclusive)
            end_year: Filter specimens up to this year (inclusive)
        """
        self.data_manager = data_manager
        self.start_year = start_year
        self.end_year = end_year

        logger.info(
            f"TaxonomyHierarchy initialized with time filter: "
            f"start_year={start_year}, end_year={end_year}"
        )

    def get_relevant_taxa(self) -> Set[str]:
        """
        Get all taxa that have specimen images in the filtered time period.

        Returns:
            Set of taxon IDs (strings) that have at least one specimen

        Example:
            >>> hierarchy = TaxonomyHierarchy(start_year=2020)
            >>> taxa = hierarchy.get_relevant_taxa()
            >>> print(f"Found {len(taxa)} taxa with specimens")
        """
        if self.data_manager is None:
            logger.warning("No data manager provided, returning empty set")
            return set()

        # Get all specimens with manual determination
        specimens = self.data_manager.get_specimens(
            start_year=self.start_year,
            end_year=self.end_year
        )

        # Extract unique taxon IDs
        # Handle both SQLAlchemy objects and dictionaries
        relevant_taxa = set()
        for specimen in specimens:
            if hasattr(specimen, 'manual_determination'):
                # SQLAlchemy object
                if specimen.manual_determination:
                    relevant_taxa.add(specimen.manual_determination)
            elif isinstance(specimen, dict) and specimen.get('determination_manual'):
                # Dictionary
                relevant_taxa.add(specimen['determination_manual'])

        logger.info(f"Found {len(relevant_taxa)} relevant taxa")
        return relevant_taxa

    def build_hierarchy(self, relevant_taxa: Set[str]) -> Dict:
        """
        Build hierarchical tree structure from taxa.

        Creates a nested dictionary where each taxon has:
        - children: Set of child taxon IDs
        - images: List of specimen image IDs at this exact level
        - parent_id: Parent taxon ID

        Args:
            relevant_taxa: Set of taxon IDs to include

        Returns:
            Dictionary with complete taxonomy tree

        Example:
            >>> taxa_set = {"Coleoptera", "Insecta", "Arthropoda"}
            >>> tree = hierarchy.build_hierarchy(taxa_set)
            >>> print(tree["Arthropoda"]["children"])
            {"Insecta"}
        """
        # Initialize with root taxon "RT" (Arthropoda)
        hierarchy = {
            "RT": {
                "children": set(),
                "images": [],
                "parent_id": None
            }
        }

        def add_taxon_and_ancestors(taxon_id: str):
            """Recursively add taxon and all ancestors to hierarchy."""
            current_id = taxon_id

            while current_id and current_id != "RT":
                # Get parent from data manager
                parent_id = self.data_manager.get_parent_taxon(current_id)

                # Add current taxon if not exists
                if current_id not in hierarchy:
                    hierarchy[current_id] = {
                        "children": set(),
                        "images": [],
                        "parent_id": parent_id
                    }

                # Add parent if not exists
                if parent_id and parent_id not in hierarchy:
                    hierarchy[parent_id] = {
                        "children": set(),
                        "images": [],
                        "parent_id": None
                    }

                # Link parent to child
                if parent_id:
                    hierarchy[parent_id]["children"].add(current_id)

                # Move up the tree
                current_id = parent_id

        # Add all relevant taxa and their ancestors
        for taxon_id in relevant_taxa:
            add_taxon_and_ancestors(taxon_id)

        # Add specimen images to each taxon
        for taxon_id in relevant_taxa:
            specimens = self.data_manager.get_specimens(
                taxon_id=taxon_id,
                start_year=self.start_year,
                end_year=self.end_year
            )

            # Handle both SQLAlchemy objects and dictionaries
            image_ids = []
            for spec in specimens:
                if hasattr(spec, 'id'):
                    image_ids.append(spec.id)
                elif isinstance(spec, dict):
                    image_ids.append(spec['id'])

            hierarchy[taxon_id]["images"].extend(image_ids)

        logger.info(f"Built hierarchy with {len(hierarchy)} taxa")
        return hierarchy

    def get_taxon_info(self, taxon_id: str) -> Tuple[str, Optional[str]]:
        """
        Get taxon scientific name and parent ID.

        Args:
            taxon_id: Taxon identifier

        Returns:
            Tuple of (scientific_name, parent_id)

        Example:
            >>> name, parent = hierarchy.get_taxon_info("Coleoptera")
            >>> print(f"{name} is a child of {parent}")
            Coleoptera is a child of Insecta
        """
        if self.data_manager is None:
            return ("Unknown Taxon", None)

        taxon = self.data_manager.get_taxon(taxon_id)

        if taxon:
            return (taxon['scientific_name'], taxon.get('parent_id'))
        else:
            return ("Unknown Taxon", None)

    def get_total_images_for_taxon(
        self,
        hierarchy: Dict,
        taxon_id: str
    ) -> int:
        """
        Calculate total number of images for taxon including all descendants.

        This recursively counts images at the taxon level and all children.

        Args:
            hierarchy: Complete taxonomy tree
            taxon_id: Taxon to count images for

        Returns:
            Total number of specimen images

        Example:
            >>> total = hierarchy.get_total_images_for_taxon(tree, "Insecta")
            >>> print(f"Insecta has {total} total specimens")
        """
        if taxon_id not in hierarchy:
            return 0

        # Count images at this level
        total = len(hierarchy[taxon_id]['images'])

        # Recursively add children
        for child_id in hierarchy[taxon_id]['children']:
            total += self.get_total_images_for_taxon(hierarchy, child_id)

        return total

    def build_tree_string(
        self,
        hierarchy: Dict,
        taxon_id: str = "RT",
        level: int = 0
    ) -> str:
        """
        Build string representation of taxonomy tree.

        Args:
            hierarchy: Complete taxonomy tree
            taxon_id: Root taxon to start from (default: "RT")
            level: Indentation level (used for recursion)

        Returns:
            Formatted tree string

        Example:
            >>> tree_str = hierarchy.build_tree_string(tree)
            >>> print(tree_str)
            Arthropoda (ID: RT, Total: 5234)
                Insecta (ID: 234, Total: 4521)
                    Coleoptera (ID: 567, Total: 1234)
        """
        if taxon_id not in hierarchy:
            return ""

        # Get taxon info
        taxon_name, _ = self.get_taxon_info(taxon_id)
        total_images = self.get_total_images_for_taxon(hierarchy, taxon_id)
        direct_images = len(hierarchy[taxon_id]['images'])

        # Format current taxon
        indent = "    " * level
        output = (
            f"{indent}{taxon_name} "
            f"(ID: {taxon_id}, "
            f"Total: {total_images}, "
            f"Direct: {direct_images})\n"
        )

        # Recursively add children
        for child_id in sorted(hierarchy[taxon_id]["children"]):
            output += self.build_tree_string(hierarchy, child_id, level + 1)

        return output

    def print_tree(self, hierarchy: Dict, taxon_id: str = "RT"):
        """
        Print taxonomy tree to console.

        Args:
            hierarchy: Complete taxonomy tree
            taxon_id: Root taxon to print from (default: "RT")

        Example:
            >>> hierarchy.print_tree(tree)
            Arthropoda (ID: RT, Total: 5234)
                ...
        """
        tree_str = self.build_tree_string(hierarchy, taxon_id)
        print(tree_str)

    def save_hierarchy(self, hierarchy: Dict, output_path: Path):
        """
        Save hierarchy to JSON file.

        Args:
            hierarchy: Complete taxonomy tree
            output_path: Path to save JSON

        Example:
            >>> hierarchy.save_hierarchy(tree, Path('./output/taxonomy.json'))
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON serialization
        serializable = {}
        for taxon_id, data in hierarchy.items():
            serializable[taxon_id] = {
                'children': list(data['children']),
                'images': data['images'],
                'parent_id': data['parent_id']
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Saved hierarchy to {output_path}")

    def load_hierarchy(self, input_path: Path) -> Dict:
        """
        Load hierarchy from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Complete taxonomy tree

        Example:
            >>> tree = hierarchy.load_hierarchy(Path('./output/taxonomy.json'))
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            serializable = json.load(f)

        # Convert lists back to sets
        hierarchy = {}
        for taxon_id, data in serializable.items():
            hierarchy[taxon_id] = {
                'children': set(data['children']),
                'images': data['images'],
                'parent_id': data['parent_id']
            }

        logger.info(f"Loaded hierarchy from {input_path}")
        return hierarchy

    def build(self) -> Dict:
        """
        Build complete taxonomy hierarchy (convenience method).

        This combines get_relevant_taxa() and build_hierarchy() in one call.

        Returns:
            Complete taxonomy tree

        Example:
            >>> hierarchy = TaxonomyHierarchy(start_year=2020)
            >>> tree = hierarchy.build()
            >>> hierarchy.print_tree(tree)
        """
        logger.info("Building complete taxonomy hierarchy")

        # Get relevant taxa
        relevant_taxa = self.get_relevant_taxa()

        # Build hierarchy
        hierarchy = self.build_hierarchy(relevant_taxa)

        logger.info(
            f"Hierarchy complete: {len(hierarchy)} taxa, "
            f"{self.get_total_images_for_taxon(hierarchy, 'RT')} total specimens"
        )

        return hierarchy

    def filter_hierarchy_by_taxon(
        self,
        hierarchy: Dict,
        top_taxon_id: str
    ) -> Dict:
        """
        Extract sub-hierarchy starting from a specific taxon.

        Useful for creating training sets for specific groups
        (e.g., only Insecta instead of all Arthropoda).

        Args:
            hierarchy: Complete taxonomy tree
            top_taxon_id: Taxon to use as new root

        Returns:
            Filtered hierarchy

        Raises:
            ValueError: If taxon_id not found in hierarchy

        Example:
            >>> insecta_tree = hierarchy.filter_hierarchy_by_taxon(tree, "Insecta")
            >>> print(len(insecta_tree))  # Only Insecta and descendants
        """
        if top_taxon_id not in hierarchy:
            raise ValueError(f"Taxon ID {top_taxon_id} not found in hierarchy")

        # Initialize with top taxon
        filtered = {top_taxon_id: hierarchy[top_taxon_id]}

        def add_descendants(taxon_id: str):
            """Recursively add all descendants."""
            for child_id in hierarchy[taxon_id]['children']:
                filtered[child_id] = hierarchy[child_id]
                add_descendants(child_id)

        # Add all descendants
        add_descendants(top_taxon_id)

        logger.info(
            f"Filtered hierarchy to {top_taxon_id}: "
            f"{len(filtered)} taxa"
        )

        return filtered

    def get_hierarchy_statistics(self, hierarchy: Dict) -> Dict:
        """
        Calculate statistics about the hierarchy.

        Args:
            hierarchy: Complete taxonomy tree

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = hierarchy.get_hierarchy_statistics(tree)
            >>> print(f"Max depth: {stats['max_depth']}")
        """
        def get_depth(taxon_id: str, current_depth: int = 0) -> int:
            """Recursively calculate maximum depth."""
            if taxon_id not in hierarchy:
                return current_depth

            children = hierarchy[taxon_id]['children']
            if not children:
                return current_depth

            return max(
                get_depth(child, current_depth + 1)
                for child in children
            )

        # Count taxa by number of children
        leaf_taxa = sum(
            1 for data in hierarchy.values()
            if len(data['children']) == 0
        )

        internal_taxa = len(hierarchy) - leaf_taxa

        # Total images
        total_images = self.get_total_images_for_taxon(hierarchy, "RT")

        stats = {
            'total_taxa': len(hierarchy),
            'leaf_taxa': leaf_taxa,
            'internal_taxa': internal_taxa,
            'max_depth': get_depth("RT"),
            'total_images': total_images,
            'avg_images_per_taxon': total_images / len(hierarchy) if hierarchy else 0
        }

        return stats
