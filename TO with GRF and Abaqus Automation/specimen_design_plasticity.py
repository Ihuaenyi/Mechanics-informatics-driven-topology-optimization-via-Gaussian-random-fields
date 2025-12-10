"""
Karhunen-Loève Expansion of GRF Heterogeneous Specimen Generator
Modified for Hill48 anisotropic plasticity criterion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.patches as patches
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')


class KLExpansionSpecimenGenerator:
    """
    Karhunen-Loève expansion based heterogeneous specimen generator
    Modified for 2D anisotropic elements
    """
    
    def __init__(self, width=20, height=40, resolution=800, design_margin=2.5, n_modes=25):
        """
        Initialize KL expansion specimen generator
        
        Parameters:
        -----------
        width, height : float
            Specimen dimensions [mm]
        resolution : int
            Grid resolution
        design_margin : float
            Margin from specimen edges [mm]
        n_modes : int
            Number of KL modes (controls pattern complexity)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.design_margin = design_margin
        self.n_modes = n_modes
        
        # Create coordinate grids
        self.x = np.linspace(0, width, resolution)
        self.y = np.linspace(0, height, resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Define design region
        self.design_region = {
            'x_min': design_margin,
            'x_max': width - design_margin,
            'y_min': design_margin,
            'y_max': height - design_margin,
            'width': width - 2*design_margin,
            'height': height - 2*design_margin
        }
        
        # Create design region mask
        self.design_mask = ((self.X >= self.design_region['x_min']) & 
                           (self.X <= self.design_region['x_max']) &
                           (self.Y >= self.design_region['y_min']) & 
                           (self.Y <= self.design_region['y_max']))
        
        # Initialize
        self.phi = np.ones((resolution, resolution))
        self.grf_field = None
        self.eigenvalues = None
        self.eigenfunctions = None
        self.xi_coefficients = None
        
        # Precompute eigenfunctions (Fourier basis)
        self._compute_eigenfunctions()
    
    def _compute_eigenfunctions(self):
        """
        Compute orthonormal Fourier basis functions as eigenfunctions
        """
        self.eigenfunctions = np.zeros((self.n_modes, self.resolution, self.resolution))
        
        # Generate mode indices (kx, ky pairs)
        modes_per_dim = int(np.ceil(np.sqrt(self.n_modes)))
        mode_count = 0
        
        for kx in range(1, modes_per_dim + 1):
            for ky in range(1, modes_per_dim + 1):
                if mode_count >= self.n_modes:
                    break
                
                # Normalized Fourier basis functions
                phi = (np.sqrt(4 / (self.width * self.height)) * 
                       np.sin(kx * np.pi * self.X / self.width) * 
                       np.sin(ky * np.pi * self.Y / self.height))
                
                self.eigenfunctions[mode_count] = phi
                mode_count += 1
            
            if mode_count >= self.n_modes:
                break
    
    def generate_grf_field(self, eigenvalues, mean_field=0.0, xi_coefficients=None, seed=42):
        """ Generate GRF field using KL expansion"""
        if len(eigenvalues) != self.n_modes:
            raise ValueError(f"Expected {self.n_modes} eigenvalues, got {len(eigenvalues)}")
        
        np.random.seed(seed)
        
        # Store parameters
        self.eigenvalues = np.array(eigenvalues)
        self.mean_field = mean_field
        
        # Generate or use provided random coefficients
        if xi_coefficients is None:
            self.xi_coefficients = np.random.randn(self.n_modes)
        else:
            self.xi_coefficients = np.array(xi_coefficients)
        
        # Ensure eigenvalues are non-negative
        self.eigenvalues = np.maximum(self.eigenvalues, 1e-12)
        
        # Generate field using KL expansion
        field = np.full((self.resolution, self.resolution), mean_field)
        
        for i in range(self.n_modes):
            contribution = np.sqrt(self.eigenvalues[i]) * self.eigenfunctions[i] * self.xi_coefficients[i]
            field += contribution
        
        # Normalize field to [0, 1]
        field_min = field.min()
        field_max = field.max()
        if field_max > field_min:
            field = (field - field_min) / (field_max - field_min)
        else:
            field = np.zeros_like(field)
        
        # Apply design region mask
        self.grf_field = np.where(self.design_mask, field, 1.0)
        
        return self.grf_field
    
    def apply_volume_constraint(self, target_void_fraction=0.25, max_void_fraction=0.40, 
                               corner_rounding=True, min_radius=0.5):
        """Apply volume constraint to create binary geometry with rounded corners"""
        if self.grf_field is None:
            raise ValueError("Must generate GRF field first")
        
        target_void_fraction = min(target_void_fraction, max_void_fraction)
        
        # Find threshold for target void fraction
        design_field = self.grf_field[self.design_mask]
        
        def objective(threshold):
            binary_field = (design_field < threshold).astype(float)
            actual_void_fraction = np.mean(binary_field)
            return abs(actual_void_fraction - target_void_fraction)
        
        # Optimize threshold
        result = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')
        optimal_threshold = result.x
        
        # Apply threshold
        binary_field = (self.grf_field < optimal_threshold).astype(float)
        
        # Apply corner rounding if requested
        if corner_rounding:
            binary_field = self._apply_corner_rounding(binary_field, min_radius)
        else:
            # Light smoothing to reduce pixelation
            binary_field = gaussian_filter(binary_field.astype(float), sigma=0.5)
            binary_field = (binary_field > 0.5).astype(float)
        
        # Create final geometry
        self.phi = np.where(self.design_mask, 1.0 - binary_field, 1.0)
        
        return self.phi
    
    def _apply_corner_rounding(self, binary_field, min_radius):
        """Apply corner rounding to avoid sharp angles"""
        from scipy.ndimage import binary_opening, binary_closing
        from scipy.ndimage import distance_transform_edt
        
        # Convert radius to pixels
        radius_pixels = int(min_radius * self.resolution / self.width)
        radius_pixels = max(1, radius_pixels)
        
        # Create circular structuring element
        y, x = np.ogrid[-radius_pixels:radius_pixels+1, -radius_pixels:radius_pixels+1]
        disk = x*x + y*y <= radius_pixels*radius_pixels
        
        # Apply morphological operations
        rounded_field = binary_opening(binary_field, disk)
        rounded_field = binary_closing(rounded_field, disk)
        
        # Additional Gaussian smoothing
        sigma = radius_pixels * 0.3
        final_field = gaussian_filter(rounded_field.astype(float), sigma=sigma)
        final_field = (final_field > 0.5).astype(float)
        
        return final_field
    
    def extract_contours(self, min_area=1.0, max_points=200):
        """
        Extract ABAQUS-compatible contours
        
        Parameters:
        -----------
        min_area : float
            Minimum void area [mm²]
        max_points : int
            Maximum points per contour for ABAQUS efficiency
        """
        try:
            from skimage import measure
            
            # Extract contours at 0.5 level
            contours = measure.find_contours(self.phi, 0.5)
            
            # Process contours for ABAQUS compatibility
            self.contours = []
            self.domain_boundary = None
            
            for contour in contours:
                if len(contour) < 8:  # Skip tiny contours
                    continue
                
                # Convert to physical coordinates
                physical_contour = []
                for point in contour:
                    y_pixel, x_pixel = point
                    x_phys = x_pixel * self.width / self.resolution
                    y_phys = y_pixel * self.height / self.resolution
                    physical_contour.append((x_phys, y_phys))
                
                physical_contour = np.array(physical_contour)
                
                # Check if outer boundary
                if self._is_outer_boundary(physical_contour):
                    self.domain_boundary = physical_contour
                    continue
                
                # Check minimum area
                area = self._calculate_contour_area(physical_contour)
                if area < min_area:
                    continue
                
                # Process contour for ABAQUS
                processed_contour = self._process_contour_for_abaqus(physical_contour, max_points)
                if processed_contour is not None:
                    self.contours.append(processed_contour)
            
            # Create domain boundary if not found
            if self.domain_boundary is None:
                self.domain_boundary = np.array([
                    (0, 0), (self.width, 0), (self.width, self.height), 
                    (0, self.height), (0, 0)
                ])
            
            return self.contours
            
        except ImportError:
            return self._extract_contours_alternative()
    
    def _process_contour_for_abaqus(self, contour, max_points):
        """
        Process individual contour for ABAQUS compatibility
        
        Key operations:
        1. Ensure closed loop
        2. Remove duplicate points
        3. Ensure CCW orientation for voids
        4. Limit point count
        5. Smooth if needed
        """
        if len(contour) < 3:
            return None
        
        # Remove duplicate points
        cleaned_contour = []
        for i, point in enumerate(contour):
            if i == 0 or np.linalg.norm(point - contour[i-1]) > 1e-6:
                cleaned_contour.append(point)
        
        if len(cleaned_contour) < 3:
            return None
        
        cleaned_contour = np.array(cleaned_contour)
        
        # Ensure closure
        if np.linalg.norm(cleaned_contour[0] - cleaned_contour[-1]) > 1e-6:
            cleaned_contour = np.vstack([cleaned_contour, cleaned_contour[0:1]])
        
        # Ensure CCW orientation (for voids)
        if self._is_clockwise(cleaned_contour):
            cleaned_contour = np.flip(cleaned_contour, axis=0)
        
        # Limit point count
        if len(cleaned_contour) > max_points:
            cleaned_contour = self._resample_contour(cleaned_contour, max_points)
        
        return cleaned_contour
    
    def _is_clockwise(self, contour):
        """Check if contour is clockwise oriented"""
        if len(contour) < 3:
            return False
        
        # Calculate signed area
        signed_area = 0.0
        for i in range(len(contour) - 1):
            x1, y1 = contour[i]
            x2, y2 = contour[i + 1]
            signed_area += (x2 - x1) * (y2 + y1)
        
        return signed_area > 0  # Positive = clockwise
    
    def _resample_contour(self, contour, max_points):
        """Resample contour to limit number of points"""
        if len(contour) <= max_points:
            return contour
        
        # Calculate cumulative arc length
        arc_lengths = [0]
        for i in range(1, len(contour)):
            dist = np.linalg.norm(contour[i] - contour[i-1])
            arc_lengths.append(arc_lengths[-1] + dist)
        
        arc_lengths = np.array(arc_lengths)
        total_length = arc_lengths[-1]
        
        # Create uniform sampling
        uniform_s = np.linspace(0, total_length, max_points)
        
        # Interpolate coordinates
        try:
            interp_x = interp1d(arc_lengths, contour[:, 0], kind='linear', assume_sorted=True)
            interp_y = interp1d(arc_lengths, contour[:, 1], kind='linear', assume_sorted=True)
            
            resampled_contour = np.column_stack([
                interp_x(uniform_s),
                interp_y(uniform_s)
            ])
            
            return resampled_contour
        except:
            # If interpolation fails, use simple subsampling
            indices = np.linspace(0, len(contour)-1, max_points, dtype=int)
            return contour[indices]
    
    def _is_outer_boundary(self, contour):
        """Check if contour represents specimen outer boundary"""
        tolerance = min(self.width, self.height) * 0.02
        boundary_points = 0
        
        for x, y in contour:
            if (x < tolerance or x > self.width - tolerance or 
                y < tolerance or y > self.height - tolerance):
                boundary_points += 1
        
        return boundary_points > len(contour) * 0.3
    
    def _calculate_contour_area(self, contour):
        """Calculate area using shoelace formula"""
        if len(contour) < 3:
            return 0.0
        
        x = contour[:, 0]
        y = contour[:, 1]
        
        area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        return area
    
    def _calculate_contour_perimeter(self, contour):
        """Calculate perimeter"""
        if len(contour) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(contour) - 1):
            perimeter += np.linalg.norm(contour[i+1] - contour[i])
        
        return perimeter
    
    def _extract_contours_alternative(self):
        """Alternative contour extraction if scikit-image not available"""
        # Simple rectangular approximation
        self.contours = []
        
        void_regions = (self.phi < 0.5) & self.design_mask
        
        if np.any(void_regions):
            y_indices, x_indices = np.where(void_regions)
            
            if len(x_indices) > 0:
                x_min_phys = np.min(x_indices) * self.width / self.resolution
                x_max_phys = np.max(x_indices) * self.width / self.resolution
                y_min_phys = np.min(y_indices) * self.height / self.resolution
                y_max_phys = np.max(y_indices) * self.height / self.resolution
                
                # Create rectangular contour
                rect_contour = np.array([
                    (x_min_phys, y_min_phys),
                    (x_max_phys, y_min_phys),
                    (x_max_phys, y_max_phys),
                    (x_min_phys, y_max_phys),
                    (x_min_phys, y_min_phys)
                ])
                
                self.contours.append(rect_contour)
        
        self.domain_boundary = np.array([
            (0, 0), (self.width, 0), (self.width, self.height), 
            (0, self.height), (0, 0)
        ])
        
        return self.contours
    
    def export_abaqus_script_2d_plane_stress(self, filename="kl_specimen_2d_abaqus.py"):
        """
        Export ABAQUS script for 2D plane stress elements (alternative approach)
        """
        if not hasattr(self, 'contours') or self.contours is None:
            self.extract_contours()
        
        # Calculate void fraction
        void_fraction = 1.0 - np.mean(self.phi[self.design_mask])
        
        script_lines = [
            "# -*- coding: mbcs -*-",
            "# KL Expansion Specimen - 2D Plane Stress Elements",
            "# Auto-generated ABAQUS script",
            "#",
            "from abaqus import *",
            "from abaqusConstants import *",
            "from odbAccess import *",
            "import regionToolset",
            "",
            f"# KL Specimen parameters",
            f"W = {self.height}   # Height (Y direction)",
            f"L = {self.width}  # Length (X direction)", 
            f"# Void fraction: {void_fraction:.4f}",
            "",
            "## Sketch",
            "s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', ",
            "    sheetSize=200.0)",
            "g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints",
            "s.setPrimaryObject(option=STANDALONE)",
            "",
            "# Rectangle geometry",
            "s.rectangle(point1=(0.0, 0.0), point2=(L, W))",
            "print('Rectangle created')",
            "",
            "# 2D Planar part",
            "p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=TWO_D_PLANAR, ",
            "    type=DEFORMABLE_BODY)",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "p.BaseShell(sketch=s)",
            "s.unsetPrimaryObject()",
            "print('2D part created')",
            "",
            "# Isotropic Material (for testing)",
            "mdb.models['Model-1'].Material(name='Material-1')",
            "mdb.models['Model-1'].materials['Material-1'].Density(table=((1500.0, ), ))",
            "mdb.models['Model-1'].materials['Material-1'].Elastic(table=((70000.0, 0.33), ))",
            "",
            "# Solid section for 2D",
            "mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', ",
            "    material='Material-1', thickness=1.0)",
            "",
            "# Section assignment",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "f = p.faces",
            "faces = f.getSequenceFromMask(mask=('[#1 ]', ), )",
            "region = p.Set(faces=faces, name='Set-1')",
            "p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, ",
            "    offsetType=MIDDLE_SURFACE, offsetField='', ",
            "    thicknessAssignment=FROM_SECTION)",
            "",
            "# Assembly",
            "a = mdb.models['Model-1'].rootAssembly",
            "a.DatumCsysByDefault(CARTESIAN)",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "a.Instance(name='Part-1-1', part=p, dependent=ON)",
            "",
            "# Step",
            "mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial')",
            "",
            "# Boundary conditions - Bottom edge fixed",
            "try:",
            "    a = mdb.models['Model-1'].rootAssembly",
            "    e1 = a.instances['Part-1-1'].edges",
            "    edges1 = e1.findAt(((L/2, 0.0, 0.0), ))",
            "    region = a.Set(edges=[edges1], name='Set-Bottom')",
            "    mdb.models['Model-1'].EncastreBC(name='BC-Bottom', createStepName='Step-1', ",
            "        region=region, localCsys=None)",
            "    print('Bottom BC applied')",
            "except:",
            "    print('Bottom BC failed')",
            "",
            "# Top edge displacement",
            "try:",
            "    a = mdb.models['Model-1'].rootAssembly",
            "    e1 = a.instances['Part-1-1'].edges", 
            "    edges1 = e1.findAt(((L/2, W, 0.0), ))",
            "    region = a.Set(edges=[edges1], name='Set-Top')",
            "    mdb.models['Model-1'].DisplacementBC(name='BC-Top', createStepName='Step-1', ",
            "        region=region, u1=0.0, u2=0.05, ur3=UNSET, ",
            "        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', ",
            "        localCsys=None)",
            "    print('Top BC applied')",
            "except:",
            "    print('Top BC failed')",
            "",
            "# Mesh",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "elemType1 = mesh.ElemType(elemCode=CPS4R, elemLibrary=STANDARD, ",
            "    secondOrderAccuracy=OFF, distortionControl=DEFAULT)",
            "elemType2 = mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD)",
            "f = p.faces",
            "faces = f.getSequenceFromMask(mask=('[#1 ]', ), )",
            "pickedRegions =(faces, )",
            "p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))",
            "p.seedPart(size=1.0, deviationFactor=0.1, minSizeFactor=0.1)",
            "p.generateMesh()",
            "print('Mesh generated')",
            "",
            "# Regenerate assembly",
            "a1 = mdb.models['Model-1'].rootAssembly",
            "a1.regenerate()",
            "",
            "# Job",
            "print('Creating job...')",
            "mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, ",
            "    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, ",
            "    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, ",
            "    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, ",
            "    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', ",
            "    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, ",
            "    numDomains=1, numGPUs=0)",
            "",
            "print('Submitting job...')",
            "mdb.jobs['Job-1'].submit(consistencyChecking=OFF)",
            "mdb.jobs['Job-1'].waitForCompletion()",
            "print('Job completed')",
            "",
            "# Post-processing",
            "try:",
            "    odb = openOdb(path='Job-1.odb')",
            "    step = odb.steps['Step-1']",
            "    frame = step.frames[-1]",
            "    stress = frame.fieldOutputs['S']",
            "    stressValues = stress.values",
            "    ",
            "    with open('stress.txt', 'w') as f:",
            "        for v in stressValues:",
            "            f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                    (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "    print('Stress data extracted successfully')",
            "    odb.close()",
            "except Exception as e:",
            "    print('Post-processing failed: %s' % str(e))"
        ]
        
        # Write file
        with open(filename, 'w') as f:
            f.write('\n'.join(script_lines))
        
        return filename

    def export_abaqus_script(self, filename="kl_specimen_abaqus.py"):
        """
        Export ABAQUS script for 2D plane stress anisotropic lamina - Based on working 3D template
        """
        if not hasattr(self, 'contours') or self.contours is None:
            self.extract_contours()
        
        # Calculate void fraction
        void_fraction = 1.0 - np.mean(self.phi[self.design_mask])
        
        script_lines = [
            "# -*- coding: mbcs -*-",
            "# KL Expansion Specimen - 2D Plane Stress Anisotropic Lamina",
            "# Based on working 3D template, converted to 2D",
            "#",
            "from abaqus import *",
            "from abaqusConstants import *",
            "from odbAccess import *",
            "#from parameters import *",
            "import regionToolset",
            "",
            f"# KL Specimen parameters",
            f"W = {self.height}   # Width (Y direction)",
            f"L = {self.width}  # Length (X direction)", 
            f"# Void fraction: {void_fraction:.4f}",
            f"# Number of KL voids: {len(self.contours)}",
            "",
            "## Sketch",
            "s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', ",
            "    sheetSize=200.0)",
            "g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints",
            "s.setPrimaryObject(option=STANDALONE)",
            "",
            "#Rectangle",
            "s.rectangle(point1=(0.0, 0.0), point2=(L, W))",
            ""
        ]
        
        # Add KL-generated void contours (if any)
        for i, contour in enumerate(self.contours):
            area = self._calculate_contour_area(contour)
            script_lines.extend([
                f"# KL Void {i+1} (Area: {area:.2f} mm²)",
                f"kl_void_{i+1}_points = ["
            ])
            
            for x, y in contour:
                script_lines.append(f"    ({x:.6f}, {y:.6f}),")
            
            script_lines.extend([
                "]",
                f"s.Spline(points=kl_void_{i+1}_points)",
                ""
            ])
        
        # Continue with 2D part creation (adapted from your 3D template)
        script_lines.extend([
            "#2D part (adapted from your working 3D template)",
            "p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=TWO_D_PLANAR, ",
            "    type=DEFORMABLE_BODY)",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "p.BaseShell(sketch=s)",
            "s.unsetPrimaryObject()",
            "",
            "#Anisotropic Material (Lamina) - Conservative properties",
            "mdb.models['Model-1'].Material(name='Material-1')",
            "mdb.models['Model-1'].materials['Material-1'].Density(table=((2.71e-06, ), ))",
            "mdb.models['Model-1'].materials['Material-1'].Elastic(type=LAMINA, table=(",
            "    (100.0, 180.0, 0.25, 60.0, 60.0, 30.0), ))",  # More conservative anisotropic properties
            "",
            "#Section for 2D - using SolidSection instead of ShellSection",
            "mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', ",
            "    material='Material-1', thickness=1.0)",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "f = p.faces",
            "faces = f.getSequenceFromMask(mask=('[#1 ]', ), )",
            "region = p.Set(faces=faces, name='Set-1')",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, ",
            "    offsetType=MIDDLE_SURFACE, offsetField='', ",
            "    thicknessAssignment=FROM_SECTION)",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "f = p.faces",
            "faces = f.getSequenceFromMask(mask=('[#1 ]', ), )",
            "region = regionToolset.Region(faces=faces)",
            "orientation=None",
            "mdb.models['Model-1'].parts['Part-1'].MaterialOrientation(region=region, ",
            "    orientationType=GLOBAL, axis=AXIS_1, additionalRotationType=ROTATION_NONE, ",
            "    localCsys=None, fieldName='', stackDirection=STACK_3)",
            "",
            "#instance",
            "a = mdb.models['Model-1'].rootAssembly",
            "a.DatumCsysByDefault(CARTESIAN)",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "a.Instance(name='Part-1-1', part=p, dependent=ON)",
            "",
            "#step - using your working step parameters",
            "mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial')",
            "mdb.models['Model-1'].steps['Step-1'].setValues(initialInc=0.05, minInc=1e-08)",
            "",
            "# AUTOMATIC EDGE MASK DETECTION FOR 2D (adapted from your 3D face detection)",
            "def find_bottom_top_edge_masks(assembly, instance_name, width=W):",
            "    \"\"\"",
            "    Automatically find the correct edge masks for bottom and top edges",
            "    Bottom edge: y = 0 (all vertices have y-coordinate = 0)",
            "    Top edge: y = W (all vertices have y-coordinate = W)",
            "    \"\"\"",
            "    instance = assembly.instances[instance_name]",
            "    edges = instance.edges",
            "    ",
            "    bottom_edge_indices = []",
            "    top_edge_indices = []",
            "    ",
            "    for i, edge in enumerate(edges):",
            "        # Get all vertices of this edge",
            "        vertices = edge.getVertices()",
            "        ",
            "        # Get Y coordinates of all vertices",
            "        y_coords = []",
            "        for vertex_id in vertices:",
            "            vertex = instance.vertices[vertex_id]",
            "            y_coord = vertex.pointOn[0][1]  # Y coordinate",
            "            y_coords.append(y_coord)",
            "        ",
            "        # Check if this edge is on bottom (all vertices at y=0)",
            "        if all(abs(y) < 1e-6 for y in y_coords):",
            "            bottom_edge_indices.append(i)",
            "            ",
            "        # Check if this edge is on top (all vertices at y=W)",
            "        elif all(abs(y - width) < 1e-6 for y in y_coords):",
            "            top_edge_indices.append(i)",
            "    ",
            "    return bottom_edge_indices, top_edge_indices",
            "",
            "def create_edge_mask(edge_indices):",
            "    \"\"\"",
            "    Create an edge mask string from a list of edge indices",
            "    ABAQUS uses hex notation where each bit represents an edge",
            "    \"\"\"",
            "    if not edge_indices:",
            "        return None",
            "    ",
            "    # Convert edge indices to a bitmask",
            "    mask_value = 0",
            "    for edge_idx in edge_indices:",
            "        mask_value |= (1 << edge_idx)",
            "    ",
            "    # Convert to hex string in ABAQUS format",
            "    mask_hex = hex(mask_value)[2:].upper()  # Remove '0x' prefix and make uppercase",
            "    mask_string = \"[#\" + mask_hex + \" ]\"",
            "    ",
            "    return mask_string",
            "",
            "# Find bottom and top edge indices",
            "bottom_indices, top_indices = find_bottom_top_edge_masks(a, 'Part-1-1', width=W)",
            "",
            "# Create masks",
            "bottom_mask = create_edge_mask(bottom_indices)",
            "top_mask = create_edge_mask(top_indices)",
            "",
            "# Apply boundary conditions using the automatically determined masks",
            "",
            "# Bottom edge - Fixed boundary condition",
            "if bottom_mask and bottom_indices:",
            "    try:",
            "        a = mdb.models['Model-1'].rootAssembly",
            "        e1 = a.instances['Part-1-1'].edges",
            "        edges1 = e1.getSequenceFromMask(mask=(bottom_mask, ), )",
            "        region = a.Set(edges=edges1, name='Set-Bottom-Auto')",
            "        mdb.models['Model-1'].EncastreBC(name='BC-Bottom-Auto', createStepName='Step-1', ",
            "            region=region, localCsys=None)",
            "    except:",
            "        pass",
            "",
            "# Top edge - Displacement boundary condition (reduced from 2.0 to 0.2)",
            "if top_mask and top_indices:",
            "    try:",
            "        a = mdb.models['Model-1'].rootAssembly",
            "        e1 = a.instances['Part-1-1'].edges",
            "        edges1 = e1.getSequenceFromMask(mask=(top_mask, ), )",
            "        region = a.Set(edges=edges1, name='Set-Top-Auto')",
            "        mdb.models['Model-1'].DisplacementBC(name='BC-Top-Auto', createStepName='Step-1', ",
            "            region=region, u1=0.0, u2=0.5, ur3=0.0, ",
            "            amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', ",
            "            localCsys=None)",
            "    except:",
            "        pass",
            "",
            "#mesh - using your working mesh parameters",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "p.seedPart(size=0.5, deviationFactor=0.1, minSizeFactor=0.1)",
            "p = mdb.models['Model-1'].parts['Part-1']",
            "p.generateMesh()",
            "a1 = mdb.models['Model-1'].rootAssembly",
            "a1.regenerate()",
            "a = mdb.models['Model-1'].rootAssembly",
            "",
            "# Job - using your working job parameters",
            "mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, ",
            "    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, ",
            "    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, ",
            "    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, ",
            "    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', ",
            "    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, ",
            "    numDomains=4, numGPUs=0)",
            "mdb.jobs['Job-1'].submit(consistencyChecking=OFF)",
            "",
            "# ------------- POST PROCESSING (adapted from your working template) ---------------------------",
            "from odbAccess import *",
            "from abaqusConstants import *",
            "import numpy as np",
            "",
            "# Open the ODB file",
            "odb = openOdb(path='Job-1.odb')",
            "",
            "# Access the step",
            "step = odb.steps['Step-1']",
            "",
            "# Get the last frame of the step",
            "frame1 = step.frames[-1]",
            "",
            "frame2 = step.frames[-2]",
            "",
            "frame3 = step.frames[-3]",
            "",
            "frame4 = step.frames[-4]",
            "",
            "frame5 = step.frames[-5]",
            "",
            "# Extract the stress fields for all elements (2D plane stress: S11, S22, S12, S33)",
            "stress1 = frame1.fieldOutputs['S']",
            "stressValues1 = stress1.values",
            "",
            "stress2 = frame2.fieldOutputs['S']",
            "stressValues2 = stress2.values",
            "",
            "stress3 = frame3.fieldOutputs['S']",
            "stressValues3 = stress3.values",
            "",
            "stress4 = frame4.fieldOutputs['S']",
            "stressValues4 = stress4.values",
            "",
            "stress5 = frame5.fieldOutputs['S']",
            "stressValues5 = stress5.values",
            "",
            "# Extract the strain fields for all elements (2D plane stress: E11, E22, E12, E33)",
            "strain1 = frame1.fieldOutputs['E']",
            "strainValues1 = strain1.values",
            "",
            "strain2 = frame2.fieldOutputs['E']",
            "strainValues2 = strain2.values",
            "",
            "strain3 = frame3.fieldOutputs['E']",
            "strainValues3 = strain3.values",
            "",
            "strain4 = frame4.fieldOutputs['E']",
            "strainValues4 = strain4.values",
            "",
            "strain5 = frame5.fieldOutputs['E']",
            "strainValues5 = strain5.values",
            "",
            "# Write stress data to file (4 components for 2D plane stress)",
            "with open('stress1.txt', 'w') as f:",
            "    for v in stressValues1:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "with open('stress2.txt', 'w') as f:",
            "    for v in stressValues2:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "with open('stress3.txt', 'w') as f:",
            "    for v in stressValues3:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "with open('stress4.txt', 'w') as f:",
            "    for v in stressValues4:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "with open('stress5.txt', 'w') as f:",
            "    for v in stressValues5:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "# Write strain data to file (4 components for 2D plane stress)",
            "with open('strain1.txt', 'w') as f:",
            "    for v in strainValues1:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "with open('strain2.txt', 'w') as f:",
            "    for v in strainValues2:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "with open('strain3.txt', 'w') as f:",
            "    for v in strainValues3:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "with open('strain4.txt', 'w') as f:",
            "    for v in strainValues4:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "with open('strain5.txt', 'w') as f:",
            "    for v in strainValues5:",
            "        f.write('%.4e, %.4e, %.4e, %.4e \\n' % ",
            "                (v.data[0], v.data[1], v.data[2], v.data[3]))",
            "",
            "#Close the ODB file",
            "odb.close()"
        ])
        
        # Write file
        with open(filename, 'w') as f:
            f.write('\n'.join(script_lines))
        
        return filename

    def analyze_eigenvalue_contributions(self):
        """Analyze how different eigenvalues contribute to the final pattern"""
        if self.eigenvalues is None:
            return
        
        total_contribution = np.sum(np.sqrt(self.eigenvalues))
        modes_per_dim = int(np.ceil(np.sqrt(self.n_modes)))
        
        contributions = []
        mode_count = 0
        for kx in range(1, modes_per_dim + 1):
            for ky in range(1, modes_per_dim + 1):
                if mode_count >= self.n_modes:
                    break
                
                eigenval = self.eigenvalues[mode_count]
                contribution = np.sqrt(eigenval) / total_contribution * 100
                char_length = min(self.width / kx, self.height / ky)
                
                contributions.append({
                    'mode': mode_count + 1,
                    'kx': kx,
                    'ky': ky,
                    'eigenvalue': eigenval,
                    'contribution': contribution,
                    'char_length': char_length
                })
                
                mode_count += 1
            
            if mode_count >= self.n_modes:
                break
        
        return contributions


def create_kl_specimen(eigenvalues, target_void_fraction=0.25, 
                      mean_field=0.0, seed=42, n_modes=25,
                      corner_rounding=True, min_radius=0.5,
                      width=20, height=40, resolution=800):
    """
    Create KL specimen with ABAQUS-compatible contours
    
    Parameters:
    -----------
    eigenvalues : list or array
        Eigenvalues for KL expansion (must have n_modes elements)
    target_void_fraction : float
        Target void fraction (0.0 to 1.0)
    mean_field : float
        Mean field value for GRF
    seed : int
        Random seed for reproducibility
    n_modes : int
        Number of KL modes
    corner_rounding : bool
        Whether to apply corner rounding
    min_radius : float
        Minimum radius for corner rounding [mm]
    width, height : float
        Specimen dimensions [mm]
    resolution : int
        Grid resolution
    
    Returns:
    --------
    KLExpansionSpecimenGenerator
        Generator object with all results
    """
    # Initialize generator
    gen = KLExpansionSpecimenGenerator(width=width, height=height, 
                                      resolution=resolution, n_modes=n_modes)
    
    # Generate GRF field
    gen.generate_grf_field(eigenvalues, mean_field=mean_field, seed=seed)
    
    # Apply volume constraint
    gen.apply_volume_constraint(target_void_fraction=target_void_fraction,
                               corner_rounding=corner_rounding,
                               min_radius=min_radius)
    
    # Extract contours and export ABAQUS script
    gen.extract_contours()
    gen.export_abaqus_script()
    
    return gen
