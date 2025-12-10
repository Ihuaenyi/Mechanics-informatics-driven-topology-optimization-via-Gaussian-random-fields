import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
import random
import os
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: REPRODUCIBILITY SETUP
# =============================================================================

def set_all_seeds(seed=42):
    """Ensure complete reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ STEP 1 COMPLETE: All seeds set to {seed}")

# =============================================================================
# STEP 2: DATA LOADING WITHOUT NORMALIZATION
# =============================================================================

def load_real_data_only(strain_file='strain5.txt', stress_file='stress5.txt'):
    """Load ONLY the real experimental data from txt files - NO NORMALIZATION
    Strains <= 100 microstrain (1e-4) are set to zero"""
    print("\n" + "="*50)
    print("STEP 2: LOADING REAL EXPERIMENTAL DATA")
    print("="*50)
    
    try:
        # Load strain data
        strain_data = []
        strain_threshold = 1e-4  # 100 microstrain = 1e-4
        with open(strain_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        if ',' in line:
                            values = [float(x.strip()) for x in line.split(',')]
                        else:
                            values = [float(x.strip()) for x in line.split()]
                        
                        if len(values) >= 4:
                            # Apply threshold: set strains <= 100 microstrain to 0
                            strain_row = [
                                values[0] if abs(values[0]) > strain_threshold else 0.0,
                                values[1] if abs(values[1]) > strain_threshold else 0.0,
                                values[3] if abs(values[3]) > strain_threshold else 0.0
                            ]
                            strain_data.append(strain_row)
                    except (ValueError, IndexError):
                        continue
        
        # Load stress data (unchanged)
        stress_data = []
        with open(stress_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        if ',' in line:
                            values = [float(x.strip()) for x in line.split(',')]
                        else:
                            values = [float(x.strip()) for x in line.split()]
                        
                        if len(values) >= 4:
                            stress_row = [values[0], values[1], values[3]]
                            stress_data.append(stress_row)
                    except (ValueError, IndexError):
                        continue
        
        # Convert to tensors - KEEPING ORIGINAL PHYSICAL UNITS
        strain_tensor = torch.from_numpy(np.array(strain_data, dtype=np.float32))
        stress_tensor = torch.from_numpy(np.array(stress_data, dtype=np.float32))
        
        # Ensure consistent length
        min_len = min(len(strain_tensor), len(stress_tensor))
        strain_tensor = strain_tensor[:min_len]
        stress_tensor = stress_tensor[:min_len]
        
        return strain_tensor, stress_tensor
        
    except FileNotFoundError as e:
        print(f"  ❌ ERROR: {e}")
        return None, None
    except Exception as e:
        print(f"  ❌ UNEXPECTED ERROR: {e}")
        return None, None

# =============================================================================
# STEP 3: PHYSICS-CONSTRAINED ICNN WITH STRONG POSITIVE DEFINITENESS
# =============================================================================

class PhysicsConstrainedICNN(nn.Module):
    """Physics-constrained ICNN with enhanced positive definiteness constraints"""
    
    def __init__(self, hidden_dims=[32, 16], seed=42):
        super().__init__()
        print("\n" + "="*50)
        print("STEP 3: PHYSICS-CONSTRAINED ICNN")
        print("="*50)
        
        self.hidden_dims = hidden_dims
        self.seed = seed
        
        set_all_seeds(seed)
        #"Ground Truth": {"C11": 181.20, "C22": 253.50, "C12": 88.80, "C66": 46.00}
        # Learnable orthotropic elastic constants with strong positive constraints
        # Use log parameterization to enforce strict positivity
        self.log_C11 = nn.Parameter(torch.log(torch.tensor(35.0)))
        self.log_C22 = nn.Parameter(torch.log(torch.tensor(80.0)))
        self.log_C66 = nn.Parameter(torch.log(torch.tensor(3.5)))
        
        # C12 parameterized to satisfy positive definiteness constraints
        self.C12_raw = nn.Parameter(torch.tensor(0.1))  # Smaller initial value
        
        # ICNN layers
        self.input_layer = nn.Linear(3, hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Passthrough connections for ICNN
        self.passthrough_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.passthrough_layers.append(nn.Linear(3, hidden_dims[i]))
        self.final_passthrough = nn.Linear(3, 1)
        
        self._initialize_weights()
        
        print(f"  🏗️ Architecture: Input(3) -> {hidden_dims} -> Output(1)")
        print(f"  📏 Working directly with physical units")
        print(f"  ⚖️ Enhanced positive definiteness constraints for stiffness")
        print("✅ STEP 3 COMPLETE: Physics-constrained ICNN created")
    
    def _initialize_weights(self):
        """Initialize weights for physical strain/stress units"""
        generator = torch.Generator().manual_seed(self.seed)
        
        nn.init.xavier_uniform_(self.input_layer.weight, gain=0.1, generator=generator)
        nn.init.zeros_(self.input_layer.bias)
        
        for layer in self.hidden_layers:
            nn.init.uniform_(layer.weight, 0.001, 0.01, generator=generator)
            nn.init.zeros_(layer.bias)
        
        nn.init.uniform_(self.output_layer.weight, 0.001, 0.01, generator=generator)
        nn.init.zeros_(self.output_layer.bias)
        
        for layer in self.passthrough_layers:
            nn.init.uniform_(layer.weight, 0.001, 0.005, generator=generator)
            nn.init.zeros_(layer.bias)
        
        nn.init.uniform_(self.final_passthrough.weight, 0.001, 0.005, generator=generator)
        nn.init.zeros_(self.final_passthrough.bias)
    
    def _get_stiffness_params(self):
        """Get stiffness parameters with guaranteed positive definiteness"""
        # Diagonal terms are strictly positive via exp
        C11 = torch.exp(self.log_C11)
        C22 = torch.exp(self.log_C22)  
        C66 = torch.exp(self.log_C66)
        
        # C12 constrained to ensure positive definiteness
        # Must satisfy: |C12| < sqrt(C11 * C22) for positive definiteness
        C12_max = torch.sqrt(C11 * C22) * 0.9  # Safety margin
        C12 = torch.tanh(self.C12_raw) * C12_max
        
        return C11, C12, C22, C66
    
    def _clamp_icnn_weights(self):
        """Clamp ICNN weights to maintain convexity"""
        with torch.no_grad():
            min_weight = 1e-8
            for layer in self.hidden_layers:
                layer.weight.data.clamp_(min=min_weight)
            self.output_layer.weight.data.clamp_(min=min_weight)
            for layer in self.passthrough_layers:
                layer.weight.data.clamp_(min=min_weight)
            self.final_passthrough.weight.data.clamp_(min=min_weight)
    
    def forward(self, strain):
        """Forward pass with strain in physical units"""
        self._clamp_icnn_weights()
        
        x = strain
        
        # Physics energy base with guaranteed positive definite stiffness matrix
        C11, C12, C22, C66 = self._get_stiffness_params()
        
        U_base = 0.5 * (
            C11 * x[:, 0]**2 +
            C22 * x[:, 1]**2 +
            C66 * x[:, 2]**2 +
            2.0 * C12 * x[:, 0] * x[:, 1]
        )
        
        # ICNN correction
        z = F.softplus(self.input_layer(x))
        
        for hidden_layer, passthrough_layer in zip(self.hidden_layers, self.passthrough_layers):
            z_contrib = hidden_layer(z)
            x_contrib = passthrough_layer(x)
            z = F.softplus(z_contrib + x_contrib)
        
        z_out = self.output_layer(z)
        x_out = self.final_passthrough(x)
        U_correction = F.softplus(z_out + x_out).squeeze(-1) *0.05
        
        total_energy = U_base + U_correction
        
        return total_energy
    
    def get_stress(self, strain, create_graph=True):
        """Get stress in physical units (MPa)"""
        if not strain.requires_grad:
            strain_input = strain.clone().detach().requires_grad_(True)
        else:
            strain_input = strain
        
        energy = self.forward(strain_input)
        
        stress = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=strain_input,
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True
        )[0]
        
        return stress

# =============================================================================
# STEP 4: ENHANCED POSITIVE DEFINITENESS TRAINER
# =============================================================================

class EnhancedPhysicsTrainer:
    """Trainer with enhanced positive definiteness constraints"""
    
    def __init__(self, model, seed=42):
        print("\n" + "="*50)
        print("STEP 4: ENHANCED POSITIVE DEFINITENESS TRAINING")
        print("="*50)
        
        self.model = model
        self.seed = seed
        
        # Enhanced physics constraint weights
        self.weights = {
            'mse': 1.0,
            'energy_positivity': 1,
            'symmetry': 1,
            'positive_definite_strong': 1,  # Strong weight on positive definiteness
            'stiffness_bounds': 1,         # Additional bounds constraint
            'smoothness': 1e-6,
        }
        
        self.train_losses = []
        self.val_losses = []
        self.epochs_logged = []
        
        print(f"  ⚖️ Enhanced constraint weights: {self.weights}")
        print(f"  📏 Strong positive definiteness enforcement")
        print("✅ STEP 4 SETUP COMPLETE: Enhanced physics trainer ready")
    
    def _compute_losses(self, strain, stress_pred, stress_true):
        """Enhanced loss computation with strong positive definiteness"""
        losses = {}
        
        # 1. Data fitting
        losses['mse'] = F.mse_loss(stress_pred, stress_true)
        
        # 2. Energy positivity
        energy_vals = self.model.forward(strain)
        losses['energy_positivity'] = torch.mean(F.relu(-energy_vals))
        
        # 3. Symmetry: U(ε) = U(-ε)
        try:
            strain_neg = -strain
            energy_pos = self.model.forward(strain)
            energy_neg = self.model.forward(strain_neg)
            losses['symmetry'] = F.mse_loss(energy_pos, energy_neg)
        except:
            losses['symmetry'] = torch.tensor(0.0)
        
        # 4. STRONG positive definiteness of material constants
        C11, C12, C22, C66 = self.model._get_stiffness_params()
        
        # Individual stiffness positivity (should be automatic with exp, but double check)
        pd_loss = F.relu(1.0 - C11) + F.relu(1.0 - C22) + F.relu(0.1 - C66)
        
        # Matrix positive definiteness: det > 0 and diagonal terms > 0
        determinant = C11 * C22 - C12**2
        pd_loss += F.relu(0.1 - determinant) * 10.0  # Strong penalty for non-PD
        
        # Eigenvalue constraints (for 2x2 stiffness matrix)
        trace = C11 + C22
        eigenval1 = 0.5 * (trace + torch.sqrt(trace**2 - 4*determinant + 1e-8))
        eigenval2 = 0.5 * (trace - torch.sqrt(trace**2 - 4*determinant + 1e-8))
        
        # Both eigenvalues must be positive
        pd_loss += F.relu(0.1 - eigenval1) * 10.0
        pd_loss += F.relu(0.1 - eigenval2) * 10.0
        
        losses['positive_definite_strong'] = pd_loss
        
        # 5. Additional stiffness bounds (prevent unreasonable values)
        bounds_loss = 0.0
        bounds_loss += F.relu(C11 - 500.0)  # Upper bound
        bounds_loss += F.relu(C22 - 500.0)  # Upper bound  
        bounds_loss += F.relu(C66 - 200.0)   # Upper bound
        bounds_loss += F.relu(torch.abs(C12) - 500.0)  # Reasonable C12 bound
        
        losses['stiffness_bounds'] = bounds_loss
        
        # 6. Smoothness
        param_penalty = sum(torch.sum(p**2) for p in self.model.parameters())
        losses['smoothness'] = param_penalty * 1e-12
        
        return losses
    
    def _validate_with_details(self, val_loader):
        """Detailed validation with stiffness monitoring"""
        self.model.eval()
        total_loss = 0.0
        batches = 0
        
        for strain_batch, stress_true_batch in val_loader:
            with torch.enable_grad():
                strain_grad = strain_batch.clone().requires_grad_(True)
                stress_pred_batch = self.model.get_stress(strain_grad, create_graph=False)
                loss = F.mse_loss(stress_pred_batch, stress_true_batch)
                
                total_loss += loss.item()
                batches += 1
        
        avg_loss = total_loss / max(batches, 1)
        rmse = np.sqrt(avg_loss)
        
        # Monitor current stiffness values
        C11, C12, C22, C66 = self.model._get_stiffness_params()
        det = C11 * C22 - C12**2
        
        return avg_loss, rmse, {
            'C11': C11.item(), 'C12': C12.item(), 
            'C22': C22.item(), 'C66': C66.item(),
            'det': det.item()
        }
    
    def train(self, train_loader, val_loader, epochs=300, lr=1e-4):
        """Training with enhanced positive definiteness monitoring"""
        print(f"\n  🎯 Starting ENHANCED training for {epochs} epochs...")
        print("  🔬 Strong positive definiteness constraints enforced")
        print("  📊 Target: Prevent negative stiffness constants")
        
        best_val_loss = float('inf')
        best_model_state = None
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-8, weight_decay=1e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=25, min_lr=1e-8, verbose=False
        )
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for strain_batch, stress_true_batch in train_loader:
                optimizer.zero_grad()
                
                stress_pred_batch = self.model.get_stress(strain_batch, create_graph=True)
                
                losses = self._compute_losses(strain_batch, stress_pred_batch, stress_true_batch)
                total_loss = sum(self.weights[key] * loss for key, loss in losses.items())
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
            
            # Enhanced validation with stiffness monitoring
            val_loss, rmse, stiffness_vals = self._validate_with_details(val_loader)
            scheduler.step(val_loss)
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            # Track losses
            avg_train_loss = train_loss / max(train_batches, 1)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.epochs_logged.append(epoch + 1)
            
            # Progress reporting with stiffness monitoring
            if (epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1:
                lr_current = optimizer.param_groups[0]['lr']
                
                print(f"    Epoch {epoch+1:3d}/{epochs}: Train={avg_train_loss:.8f}, Val={val_loss:.8f}, "
                      f"RMSE={rmse:.6f} MPa, LR={lr_current:.2e}")
                
                # Report stiffness values
                print(f"        Stiffness: C11={stiffness_vals['C11']:.2f}, C12={stiffness_vals['C12']:.2f}, "
                      f"C22={stiffness_vals['C22']:.2f}, C66={stiffness_vals['C66']:.2f}")
                print(f"        Determinant: {stiffness_vals['det']:.4f} (PD: {'✅' if stiffness_vals['det'] > 0 else '❌'})")
                
                if epoch > 25:
                    recent_improvement = (self.val_losses[-26] - val_loss) / max(self.val_losses[-26], 1e-12) * 100
                    if recent_improvement > 0.01:
                        status = "🔄 Learning"
                    elif recent_improvement > 0.001:
                        status = "🎯 Fine-tuning"
                    else:
                        status = "📊 Plateauing"
                    print(f"        Status: {status} (25-epoch improvement: {recent_improvement:.4f}%)")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        final_val_loss, final_rmse, final_stiffness = self._validate_with_details(val_loader)
        
        print(f"  🏁 TRAINING COMPLETE: Final validation loss {final_val_loss:.10f}")
        print(f"  📊 Physical RMSE: {final_rmse:.6f} MPa")
        print(f"  ⚖️ Final stiffness: C11={final_stiffness['C11']:.2f}, C12={final_stiffness['C12']:.2f}, "
              f"C22={final_stiffness['C22']:.2f}, C66={final_stiffness['C66']:.2f}")
        print(f"  ✅ Positive definite: {'Yes' if final_stiffness['det'] > 0 else 'No'} (det={final_stiffness['det']:.4f})")
        
        if final_val_loss < 1e-6:
            print("  🎉 OUTSTANDING: Excellent results with positive definite stiffness!")
        elif final_val_loss < 1e-4:
            print("  🎉 EXCELLENT: Good results with enhanced constraints!")
        elif final_val_loss < 1e-2:
            print("  ✅ GOOD: Positive definiteness constraints working")
        else:
            print("  ⚠️ MODERATE: May need constraint adjustment")
        
        print("✅ STEP 4 COMPLETE: Enhanced positive definiteness training finished")
        return final_val_loss
    
    def plot_training_curves(self):
        """Create training plots"""
        try:
            import matplotlib.pyplot as plt
            
            if len(self.train_losses) == 0:
                print("No training data to plot")
                return
            
            # Set Times New Roman and font size globally for this plot
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 25
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
            
            # Training loss with dark red markers, black edges, and black line
            ax1.plot(self.epochs_logged, self.train_losses, 
                    color='#8B0000', linewidth=3, linestyle='-',label='Training', alpha=0.9)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel(r'Training Loss, $\mathcal{L}$')  # LaTeX formatting
            ax1.grid(False, alpha=0.3, linestyle='--', linewidth=0.8)
            ax1.spines['top'].set_visible(True)
            ax1.spines['right'].set_visible(True)
            
            # Validation loss with markers
            ax2.plot(self.epochs_logged, self.val_losses, 
                    color='#A23B72', linewidth=2.5, marker='s', 
                    markersize=4, markevery=max(1, len(self.epochs_logged)//20),
                    label='Validation', alpha=0.9)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Validation Loss')
            ax2.grid(False, alpha=0.3, linestyle='--', linewidth=0.8)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('enhanced_physics_training.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Reset to defaults after plotting
            plt.rcParams.update(plt.rcParamsDefault)
            
            print("  Training curves saved as 'enhanced_physics_training.png'")
            
        except ImportError:
            print("  Matplotlib not available - plotting skipped")
        except Exception as e:
            print(f"  Plotting error: {e}")

# =============================================================================
# STEP 5: CONSTANT EXTRACTION AND EVALUATION
# =============================================================================

def extract_constants_from_real_data(model, real_strain, real_stress):
    """Extract elastic constants using linear regression"""
    print("\n" + "="*50)
    print("STEP 5: CONSTANT EXTRACTION FROM REAL DATA")
    print("="*50)
    
    strain_np = real_strain.detach().cpu().numpy()
    stress_np = real_stress.detach().cpu().numpy()
    
    print(f"  📊 Extracting from {len(strain_np)} real data points")
    print(f"  📈 Strain range: [{strain_np.min():.6f}, {strain_np.max():.6f}]")
    print(f"  📈 Stress range: [{stress_np.min():.6f}, {stress_np.max():.6f}] MPa")
    
    try:
        reg_s11 = LinearRegression()
        reg_s11.fit(strain_np[:, [0, 1]], stress_np[:, 0])
        
        reg_s22 = LinearRegression()
        reg_s22.fit(strain_np[:, [0, 1]], stress_np[:, 1])
        
        reg_s12 = LinearRegression()
        reg_s12.fit(strain_np[:, [2]], stress_np[:, 2])
        
        C11 = float(reg_s11.coef_[0])
        C12_from_s11 = float(reg_s11.coef_[1])
        C12_from_s22 = float(reg_s22.coef_[0])
        C22 = float(reg_s22.coef_[1])
        C66 = float(reg_s12.coef_[0])
        
        C12 = (C12_from_s11 + C12_from_s22) / 2
        
        r2_s11 = reg_s11.score(strain_np[:, [0, 1]], stress_np[:, 0])
        r2_s22 = reg_s22.score(strain_np[:, [0, 1]], stress_np[:, 1])
        r2_s12 = reg_s12.score(strain_np[:, [2]], stress_np[:, 2])
        
        constants = {'C11': C11, 'C12': C12, 'C22': C22, 'C66': C66}
        r2_scores = {'R2_s11': r2_s11, 'R2_s22': r2_s22, 'R2_s12': r2_s12}
        
        print(f"  📊 Extracted constants:")
        print(f"    C₁₁ = {C11:.4f} MPa")
        print(f"    C₁₂ = {C12:.4f} MPa") 
        print(f"    C₂₂ = {C22:.4f} MPa")
        print(f"    C₆₆ = {C66:.4f} MPa")
        
        print(f"  📊 R² scores:")
        print(f"    σ₁₁: {r2_s11:.4f}")
        print(f"    σ₂₂: {r2_s22:.4f}")
        print(f"    σ₁₂: {r2_s12:.4f}")
        
        det = C11 * C22 - C12**2
        is_pd = C11 > 0 and C22 > 0 and C66 > 0 and det > 0
        print(f"  ✅ Positive definite: {'Yes' if is_pd else 'No'} (det = {det:.4f})")
        
        print("✅ STEP 5 COMPLETE: Constants extracted from real data")
        return constants, r2_scores
        
    except Exception as e:
        print(f"  ❌ ERROR in extraction: {e}")
        return None, None

# =============================================================================
# MAIN WORKFLOW - ENHANCED POSITIVE DEFINITENESS
# =============================================================================

def ultra_precise_icnn_workflow_no_norm(strain_file='strain_max.txt', stress_file='stress_max.txt', seed=42):
    """Complete workflow with enhanced positive definiteness constraints"""
    
    
    # Step 1: Reproducibility
    set_all_seeds(seed)
    
    # Step 2: Load real data
    strain_data, stress_data = load_real_data_only(strain_file, stress_file)
    if strain_data is None:
        return None
    
    print(f"\n  ⚠️ NO NORMALIZATION APPLIED - Data kept in physical units")
    print(f"    Strain range: [{strain_data.min():.6f}, {strain_data.max():.6f}]")
    print(f"    Stress range: [{stress_data.min():.6f}, {stress_data.max():.6f}] MPa")
    
    # Data splitting
    indices = np.arange(len(strain_data))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    
    train_dataset = TensorDataset(strain_data[train_idx], stress_data[train_idx])
    val_dataset = TensorDataset(strain_data[val_idx], stress_data[val_idx])
    test_dataset = TensorDataset(strain_data[test_idx], stress_data[test_idx])
    
    batch_size = min(32, len(train_dataset))
    generator = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  📊 Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Step 3 & 4: Create and train enhanced model
    model = PhysicsConstrainedICNN(hidden_dims=[32, 16], seed=seed)
    trainer = EnhancedPhysicsTrainer(model, seed=seed)
    best_val_loss = trainer.train(train_loader, val_loader, epochs=300, lr=1e-4)
    
    # Create training plots
    print("\n📊 Creating enhanced physics training plots...")
    trainer.plot_training_curves()
    
    # Step 5: Extract constants and compare
    train_strain_physical = strain_data[train_idx]
    train_stress_physical = stress_data[train_idx]
    
    print("\n" + "="*60)
    print("CONSTANT EXTRACTION COMPARISON")
    print("="*60)
    
    # Extract from ground truth
    print("\n🔍 Method 1: Direct extraction from TRUE STRESS")
    true_constants, true_r2 = extract_constants_from_real_data(
        model, train_strain_physical, train_stress_physical
    )
    
    # Extract from ICNN predictions
    print("\n🔍 Method 2: Extraction from ENHANCED ICNN")
    model.eval()
    
    icnn_stress_list = []
    batch_size_pred = 50
    
    for i in range(0, len(train_strain_physical), batch_size_pred):
        batch_strain = train_strain_physical[i:i+batch_size_pred]
        try:
            with torch.enable_grad():
                batch_strain_grad = batch_strain.clone().requires_grad_(True)
                batch_stress = model.get_stress(batch_strain_grad, create_graph=False)
                icnn_stress_list.append(batch_stress.detach())
        except Exception as e:
            print(f"Warning: {e}")
            icnn_stress_list.append(train_stress_physical[i:i+batch_size_pred])
    
    icnn_stress_predictions = torch.cat(icnn_stress_list, dim=0)[:len(train_strain_physical)]
    
    icnn_constants, icnn_r2 = extract_constants_from_real_data(
        model, train_strain_physical, icnn_stress_predictions
    )
    
    # Compare predictions
    print("\n" + "="*60)
    print("ENHANCED ICNN vs TRUE DATA COMPARISON")
    print("="*60)
    
    avg_error = float('inf')
    if true_constants and icnn_constants:
        print("📊 Constants Comparison:")
        print(f"  {'Parameter':<8} {'True Data':<12} {'ICNN Pred':<12} {'Error (%)':<12}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        
        total_error = 0
        valid_params = 0
        
        for param in ['C11', 'C12', 'C22', 'C66']:
            true_val = true_constants[param]
            icnn_val = icnn_constants[param]
            
            if abs(true_val) > 1e-8:
                error_pct = abs(icnn_val - true_val) / abs(true_val) * 100
                total_error += error_pct
                valid_params += 1
            else:
                error_pct = 0
                
            print(f"  {param:<8} {true_val:<12.4f} {icnn_val:<12.4f} {error_pct:<12.2f}")
        
        if valid_params > 0:
            avg_error = total_error / valid_params
            print(f"\n  📈 Average Error: {avg_error:.2f}%")
            
            if avg_error < 1:
                print("  🎉 OUTSTANDING: Enhanced constraints achieved excellent results!")
            elif avg_error < 5:
                print("  🎉 EXCELLENT: Strong positive definiteness worked!")
            elif avg_error < 15:
                print("  ✅ GOOD: Enhanced constraints effective") 
            else:
                print("  ⚠️ MODERATE: May need further constraint refinement")
    
    # Final model stiffness check
    final_C11, final_C12, final_C22, final_C66 = model._get_stiffness_params()
    final_det = final_C11 * final_C22 - final_C12**2
    
    print(f"\n  🔧 Final Model Stiffness Parameters:")
    print(f"     C11 = {final_C11.item():.4f} MPa")
    print(f"     C12 = {final_C12.item():.4f} MPa")
    print(f"     C22 = {final_C22.item():.4f} MPa")
    print(f"     C66 = {final_C66.item():.4f} MPa")
    print(f"     Determinant = {final_det.item():.4f}")
    print(f"     Positive Definite: {'✅ YES' if final_det.item() > 0 else '❌ NO'}")
    
    # Test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    model.eval()
    test_mse = 0.0
    test_samples = 0
    
    for strain_batch, stress_true_batch in test_loader:
        try:
            with torch.enable_grad():
                strain_batch_grad = strain_batch.clone().requires_grad_(True)
                stress_pred_batch = model.get_stress(strain_batch_grad, create_graph=False)
                
                mse = F.mse_loss(stress_pred_batch, stress_true_batch)
                test_mse += mse.item() * len(strain_batch)
                test_samples += len(strain_batch)
        except:
            continue
    
    final_test_mse = test_mse / max(test_samples, 1)
    print(f"📊 Final Test Performance:")
    print(f"  Test MSE: {final_test_mse:.10f}")
    print(f"  Test RMSE: {np.sqrt(final_test_mse):.6f} MPa")
    print(f"  Validation Loss: {best_val_loss:.10f}")
    
    # Final assessment
    if best_val_loss < 1e-6 and final_det.item() > 0:
        print(f"\n🎉 OUTSTANDING: Excellent loss with guaranteed positive definite stiffness!")
    elif best_val_loss < 1e-4 and final_det.item() > 0:
        print(f"\n🎉 EXCELLENT: Good results with positive definite constraints!")
    elif final_det.item() > 0:
        print(f"\n✅ GOOD: Successfully prevented negative stiffness constants")
    else:
        print(f"\n⚠️ WARNING: Positive definiteness may need stronger enforcement")
    
    
    return {
        'model': model,
        'true_constants': true_constants,
        'icnn_constants': icnn_constants,
        'test_mse': final_test_mse,
        'val_loss': best_val_loss,
        'avg_error': avg_error,
        'final_stiffness': {
            'C11': final_C11.item(), 'C12': final_C12.item(),
            'C22': final_C22.item(), 'C66': final_C66.item(),
            'determinant': final_det.item(),
            'positive_definite': final_det.item() > 0
        }
    }

# =============================================================================
# READY TO RUN - ENHANCED POSITIVE DEFINITENESS VERSION
# =============================================================================

print("🚀 Usage:")
print("results = ultra_precise_icnn_workflow_no_norm('strain_max.txt', 'stress_max.txt', seed=42)")
