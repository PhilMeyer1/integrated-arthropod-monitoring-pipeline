"""
SQLAlchemy database models for the arthropod classification pipeline.

Translated from German to English based on GLOSSARY.md.
All table and column names now follow English conventions.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

Base = declarative_base()


class Project(Base):
    """Research project (was: Projekt)"""
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    locations = relationship("Location", back_populates="project")
    metadata_definitions = relationship("ProjectMetadataDefinition", back_populates="project")


class Location(Base):
    """Sampling location/site (was: Standort)"""
    __tablename__ = 'locations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    name = Column(String(255), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    description = Column(Text)

    # Relationships
    project = relationship("Project", back_populates="locations")
    sampling_rounds = relationship("SamplingRound", back_populates="location")
    metadata_entries = relationship("MetadataEntry", back_populates="location")


class SamplingRound(Base):
    """Collection event/round (was: Sammeldurchgang)"""
    __tablename__ = 'sampling_rounds'

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Alternative ID field (was: SammeldurchgangID) - kept for compatibility
    sampling_round_id = Column(String(50), unique=True)

    location_id = Column(Integer, ForeignKey('locations.id'), nullable=False)
    date = Column(Date)
    year = Column(Integer)  # Was: Jahr
    description = Column(Text)

    # Relationships
    location = relationship("Location", back_populates="sampling_rounds")
    sample_assignments = relationship("SampleAssignment", back_populates="sampling_round")


class SampleAssignment(Base):
    """
    Sample assignment/tray (was: Zuordnung)
    Represents a single tray with specimens from a specific size fraction.
    """
    __tablename__ = 'sample_assignments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Alternative ID field (was: ZuordnungID) - main sample identifier
    sample_id = Column(String(100), unique=True, nullable=False)

    sampling_round_id = Column(Integer, ForeignKey('sampling_rounds.id'), nullable=False)
    # Was: SammeldurchgangIDRef

    size_fraction = Column(String(50))  # Was: Groessenfraktion (e.g., 'k1', '1', '2', '7')
    tray_size = Column(String(50))  # 'large' or 'small'
    weight = Column(Float)  # Biomass in mg

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sampling_round = relationship("SamplingRound", back_populates="sample_assignments")
    composite_images = relationship("CompositeImage", back_populates="sample_assignment")


class SizeFraction(Base):
    """Size fraction definitions (was: SizeFractions)"""
    __tablename__ = 'size_fractions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), unique=True, nullable=False)  # 'k1', '1', '2', '7', etc.
    min_size_mm = Column(Float)
    max_size_mm = Column(Float)
    description = Column(String(255))


class CompositeImage(Base):
    """
    Stitched/composite image (was: stitchedImage)
    Large mosaic image created from focus-stacked tiles.
    """
    __tablename__ = 'composite_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_assignment_id = Column(Integer, ForeignKey('sample_assignments.id'), nullable=False)

    file_path = Column(String(500), nullable=False)  # Was: Dateipfad
    width = Column(Integer)
    height = Column(Integer)
    megapixels = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)

    # Relationships
    sample_assignment = relationship("SampleAssignment", back_populates="composite_images")
    single_images = relationship("SingleImage", back_populates="composite_image")


class OverviewImage(Base):
    """Overview/preview image (was: overviewImage)"""
    __tablename__ = 'overview_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    composite_image_id = Column(Integer, ForeignKey('composite_images.id'))
    file_path = Column(String(500))
    thumbnail_path = Column(String(500))


class SingleImage(Base):
    """
    Individual specimen image (was: singleImage)
    Extracted from composite image via detection + segmentation.
    """
    __tablename__ = 'single_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    composite_image_id = Column(Integer, ForeignKey('composite_images.id'), nullable=False)
    sample_assignment_id = Column(Integer, ForeignKey('sample_assignments.id'))
    # Was: zuordnungID (foreign key reference)

    # File paths
    work_images_path = Column(String(500))  # Path to extracted specimen PNG
    original_image_path = Column(String(500))

    # Detection metadata
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    detection_confidence = Column(Float)

    # Segmentation metadata
    has_mask = Column(Boolean, default=False)
    mask_path = Column(String(500))

    # Taxonomic determination
    manual_determination = Column(Integer, ForeignKey('taxa.id'))
    # Was: determination_manual or bestimmung_manual

    determined_by = Column(String(100))
    determination_date = Column(DateTime)
    determination_confidence = Column(String(50))  # e.g., 'certain', 'probable', 'uncertain'

    # Flags
    is_debris = Column(Boolean, default=False)
    exclude_from_training = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    composite_image = relationship("CompositeImage", back_populates="single_images")
    taxa_determination = relationship("Taxa", foreign_keys=[manual_determination])
    inference_results = relationship("InferenceResult", back_populates="image")


class Taxa(Base):
    """Taxonomic hierarchy (was: Taxa - kept as-is)"""
    __tablename__ = 'taxa'

    id = Column(Integer, primary_key=True, autoincrement=True)
    taxon_id = Column(String(100), unique=True, nullable=False)  # e.g., '2SPTX', 'RT'
    name = Column(String(255), nullable=False)
    rank = Column(String(50))  # kingdom, phylum, class, order, family, genus, species
    parent_id = Column(String(100), ForeignKey('taxa.taxon_id'))

    # Catalogue of Life (COL) reference
    col_id = Column(String(100))
    col_name = Column(String(255))

    # Relationships
    parent = relationship("Taxa", remote_side=[taxon_id], backref='children')
    models = relationship("Model", back_populates="taxon")


class Model(Base):
    """Trained classification model (was: Modell)"""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    taxon_id = Column(String(100), ForeignKey('taxa.taxon_id'), nullable=False)
    set_number = Column(Integer)  # Model version/set number for grouping related models

    model_path = Column(String(500), nullable=False)
    model_type = Column(String(50))  # 'classification', 'detection', 'segmentation'

    # Training metadata
    training_date = Column(DateTime, default=datetime.utcnow)
    num_training_images = Column(Integer)
    num_validation_images = Column(Integer)
    num_test_images = Column(Integer)
    num_classes = Column(Integer)

    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

    # Configuration
    base_model = Column(String(100))  # e.g., 'yolo11m-cls.pt'
    epochs = Column(Integer)
    image_size = Column(Integer)
    batch_size = Column(Integer)

    # Relationships
    taxon = relationship("Taxa", back_populates="models")


class InferenceResult(Base):
    """Classification inference result (was: InferenceResults)"""
    __tablename__ = 'inference_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('single_images.id'), nullable=False)
    model_id = Column(Integer, ForeignKey('models.id'))

    # Hierarchical classification tracking
    iteration = Column(Integer)  # Which iteration of hierarchical classification
    parent_taxon = Column(String(100))  # Taxon at start of this step
    predicted_taxon = Column(String(100), ForeignKey('taxa.taxon_id'))  # Predicted child taxon
    confidence = Column(Float)

    # Actual taxon (for test set evaluation)
    actual_taxon = Column(String(100), ForeignKey('taxa.taxon_id'))
    is_correct = Column(Boolean)

    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    image = relationship("SingleImage", back_populates="inference_results")


class ModelTaxonCount(Base):
    """Training image counts per taxon (was: ModelTaxonCount)"""
    __tablename__ = 'model_taxon_counts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    taxon_id = Column(String(100), ForeignKey('taxa.taxon_id'))

    num_train_images = Column(Integer)
    num_val_images = Column(Integer)
    num_test_images = Column(Integer)


class ExcludedIdsForTraining(Base):
    """Images excluded from training (was: ExcludedIdsForTraining)"""
    __tablename__ = 'excluded_ids_for_training'

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('single_images.id'))
    set_number = Column(Integer)  # Which model set this exclusion applies to
    reason = Column(String(255))
    excluded_at = Column(DateTime, default=datetime.utcnow)


class ProjectMetadataDefinition(Base):
    """Custom metadata field definitions for projects (was: ProjectMetadataDefinition)"""
    __tablename__ = 'project_metadata_definitions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    field_name = Column(String(100), nullable=False)
    field_type = Column(String(50))  # 'text', 'number', 'date', 'boolean'
    required = Column(Boolean, default=False)

    # Relationships
    project = relationship("Project", back_populates="metadata_definitions")


class MetadataEntry(Base):
    """Custom metadata values (was: MetadataEntry)"""
    __tablename__ = 'metadata_entries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, ForeignKey('locations.id'))
    metadata_definition_id = Column(Integer, ForeignKey('project_metadata_definitions.id'))
    value = Column(Text)

    # Relationships
    location = relationship("Location", back_populates="metadata_entries")


# Helper function to create session
def get_session(database_url: str = None):
    """
    Create a database session.

    Args:
        database_url: SQLAlchemy database URL. If None, uses default SQLite.

    Returns:
        SQLAlchemy Session instance
    """
    if database_url is None:
        database_url = "sqlite:///arthropod_pipeline.db"

    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


# Helper function to initialize database
def init_database(database_url: str = None):
    """
    Initialize database schema.

    Args:
        database_url: SQLAlchemy database URL

    Returns:
        SQLAlchemy engine
    """
    if database_url is None:
        database_url = "sqlite:///arthropod_pipeline.db"

    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine
