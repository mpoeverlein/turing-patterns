// Diffusion equation solver
class DiffusionSolver {
    constructor(nx, ny, DP, DS, k) {
        this.nx = nx;
        this.ny = ny;
        this.DP = DP;  // Diffusion coefficient of compound P
        this.DS = DS;  // Diffusion coefficient of compound S
        this.D = Math.max(DP, DS); // keep track of fastest compound for time step
        this.dx = 1.0; // grid size
        this.dy = 1.0;
        this.dt = 0.01;  // Time step (ensuring stability: dt < dx^2/(4D))
        this.k = k; // coupling between [S] and [P]
        
        this.P = Array.from({ length: nx }, () => new Float64Array(ny));
        this.P_new = Array.from({ length: nx }, () => new Float64Array(ny));
        this.S = Array.from({ length: nx }, () => new Float64Array(ny));
        this.S_new = Array.from({ length: nx }, () => new Float64Array(ny));
        
        // this.initializeHotSpot();
        this.initializeRandomDistribution();
    }

    seededRandom(seed) {
        let s = seed;
        return function() {
            s = (s * 1103515245 + 12345) & 0x7fffffff;
            return s / 0x7fffffff;
        };
    }

    initializeRandomDistribution(seed = 42) {
        const rand = this.seededRandom(seed);

        // Method 1: Completely random distribution (0 to 1)
        for (let i = 0; i < this.nx; i++) {
            for (let j = 0; j < this.ny; j++) {
                this.P[i][j] = rand();
                this.S[i][j] = rand();
            }
        }
        
        // Apply boundary conditions (edges = 0)
        for (let i = 0; i < this.nx; i++) {
            this.P[i][0] = 0;              // Top boundary
            this.P[i][this.ny-1] = 0;      // Bottom boundary
            this.S[i][0] = 0;              // Top boundary
            this.S[i][this.ny-1] = 0;      // Bottom boundary
        }
        for (let j = 0; j < this.ny; j++) {
            this.P[0][j] = 0;              // Left boundary
            this.P[this.nx-1][j] = 0;      // Right boundary
            this.S[0][j] = 0;              // Left boundary
            this.S[this.nx-1][j] = 0;      // Right boundary
        }
    }
    
    step() {
        // Perform one timestep using explicit finite difference method
        const DP = this.DP;
        const DS = this.DS;

        this.D = Math.max(DP, DS); // keep track of fastest compound for time step
        const D = this.D;
        this.setDiffusionCoefficient(D);

        const dx = this.dx;
        const dy = this.dy;
        const dt = this.dt;
        const k = this.k;
        // console.log('k =', k);

        const dx2 = dx * dx;
        const dy2 = dy * dy;
        
        // Update interior points (skip boundaries)
        for (let i = 1; i < this.nx - 1; i++) {
            for (let j = 1; j < this.ny - 1; j++) {
                // Laplacian approximation
                const Plaplacian = (this.P[i+1][j] + this.P[i-1][j] - 2*this.P[i][j]) / dx2 +
                                (this.P[i][j+1] + this.P[i][j-1] - 2*this.P[i][j]) / dy2;
                this.P_new[i][j] = this.P[i][j] + DP * dt * Plaplacian;

                const Slaplacian = (this.S[i+1][j] + this.S[i-1][j] - 2*this.S[i][j]) / dx2 +
                                (this.S[i][j+1] + this.S[i][j-1] - 2*this.S[i][j]) / dy2;
                this.S_new[i][j] = this.S[i][j] + DS * dt * Slaplacian;

                // [P] dot = k1 * [P]
                // [S] dot = k1 * [P]
                // k1 = exp(-[S]*k)
                const exponent = -k * this.S[i][j];
                const k1 = exponent < -10 ? 0 : Math.exp(exponent);
                // const k1 = Math.exp(exponent);
                const dP = k1 * this.P[i][j]; // assume dS = dP
                this.P_new[i][j] = this.P_new[i][j] + dP * dt;
                this.S_new[i][j] = this.S_new[i][j] + dP * dt;
            }
        }
        
        // Top and bottom edges (excluding corners)
        for (let j = 1; j < this.ny - 1; j++) {
            this.P_new[0][j] = this.P_new[1][j];               // Top edge
            this.P_new[this.nx-1][j] = this.P_new[this.nx-2][j]; // Bottom edge
            this.S_new[0][j] = this.S_new[1][j];               // Top edge
            this.S_new[this.nx-1][j] = this.S_new[this.nx-2][j]; // Bottom edge
        }

        // Left and right edges (excluding corners)
        for (let i = 1; i < this.nx - 1; i++) {
            this.P_new[i][0] = this.P_new[i][1];               // Left edge
            this.P_new[i][this.ny-1] = this.P_new[i][this.ny-2]; // Right edge
            this.S_new[i][0] = this.S_new[i][1];               // Left edge
            this.S_new[i][this.ny-1] = this.S_new[i][this.ny-2]; // Right edge
        }

        // Handle the four corners (average of adjacent edges for smoothness)
        this.P_new[0][0] = (this.P_new[1][0] + this.P_new[0][1]) / 2;                 // Top-left
        this.P_new[0][this.ny-1] = (this.P_new[1][this.ny-1] + this.P_new[0][this.ny-2]) / 2; // Top-right
        this.P_new[this.nx-1][0] = (this.P_new[this.nx-2][0] + this.P_new[this.nx-1][1]) / 2; // Bottom-left
        this.P_new[this.nx-1][this.ny-1] = (this.P_new[this.nx-2][this.ny-1] + this.P_new[this.nx-1][this.ny-2]) / 2; // Bottom-right
        this.S_new[0][0] = (this.S_new[1][0] + this.S_new[0][1]) / 2;                 // Top-left
        this.S_new[0][this.ny-1] = (this.S_new[1][this.ny-1] + this.S_new[0][this.ny-2]) / 2; // Top-right
        this.S_new[this.nx-1][0] = (this.S_new[this.nx-2][0] + this.S_new[this.nx-1][1]) / 2; // Bottom-left
        this.S_new[this.nx-1][this.ny-1] = (this.S_new[this.nx-2][this.ny-1] + this.S_new[this.nx-1][this.ny-2]) / 2; // Bottom-right                
        // Swap grids
        [this.P, this.P_new] = [this.P_new, this.P];
        [this.S, this.S_new] = [this.S_new, this.S];
    }
    
    getMaxVal(V) {
        let maxVal = 0;
        for (let i = 0; i < this.nx; i++) {
            for (let j = 0; j < this.ny; j++) {
                maxVal = Math.max(maxVal, V[i][j]);
            }
        }
        return maxVal;
    }

    getMaxPconc() { return this.getMaxVal(this.P); }
    getMaxSconc() { return this.getMaxVal(this.S); }

    
    reset() {
        // this.initializeHotSpot();
        this.initializeRandomDistribution();
    }
    
    setDiffusionCoefficient(D) {
        this.D = D;
        // Adjust timestep for stability (CFL condition)
        const dx2 = this.dx * this.dx;
        const maxDt = dx2 / (4 * D);
        this.dt = Math.min(0.01, maxDt * 0.8);  // Safety factor 0.8
    }
    calculateTotalMass() {
        let total = 0;
        for (let i = 0; i < this.nx; i++) {
            for (let j = 0; j < this.ny; j++) {
                total += this.P[i][j];
            }
        }
        return total;
    }
}

// Visualization
class DiffusionVisualizer {
    constructor(canvasId, solver, width, height) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.solver = solver;
        this.width = width;
        this.height = height;
        this.cellWidth = width / solver.nx;
        this.cellHeight = height / solver.ny;
        this.animationId = null;
        this.isRunning = false;
        this.stepCount = 0;
        
        // Color mapping function (temperature to RGB)
        this.colorMap = this.getHotColorMap();
    }
    
    getHotColorMap() {
        // Returns a function that maps temperature [0,1] to RGB color
        return (t) => {
            // Clamp temperature
            // t = Math.min(1.0, Math.max(0.0, t));

            // Update running min/max with smoothing
            let currentMin = Infinity;
            let currentMax = -Infinity;
            
            for (let i = 1; i < this.solver.nx-1; i++) {
                for (let j = 1; j < this.solver.ny-1; j++) {
                    const temp = this.solver.P[i][j];
                    if (temp < currentMin) currentMin = temp;
                    if (temp > currentMax) currentMax = temp;
                }
            }
            
            // Smooth the range to avoid rapid fluctuations
            // this.minTemp = this.minTemp * this.colorRangeSmoothing + currentMin * (1 - this.colorRangeSmoothing);
            // this.maxTemp = this.maxTemp * this.colorRangeSmoothing + currentMax * (1 - this.colorRangeSmoothing);
            

            // Map to [0,1]
            const range = currentMax - currentMin;
            if (range < 0.001) {
                t = 0.5;
            } else {
                t = (t - currentMin) / range;
            }

            // const gray = Math.min()
            return `rgb(${Math.floor(t*255)},${Math.floor(t*255)},${Math.floor(t*255)})`;
            
            // Define color stops: blue -> cyan -> green -> yellow -> red
            const r = Math.min(1.0, t * 4);
            const g = Math.min(1.0, t * 4 - 1);
            const b = Math.min(1.0, 4 - t * 4);
            
            return `rgb(${Math.floor(r * 255)}, ${Math.floor(g * 255)}, ${Math.floor(b * 255)})`;
        };
    }
    
    draw() {
        const solver = this.solver;
        const cellW = this.cellWidth;
        const cellH = this.cellHeight;
        
        for (let i = 1; i < solver.nx-1; i++) {
            for (let j = 1; j < solver.ny-1; j++) {
                const temperature = solver.P[i][j];
                const color = this.colorMap(temperature);
                
                this.ctx.fillStyle = color;
                this.ctx.fillRect(j * cellW, i * cellH, cellW, cellH);
            }
        }
    }
    
    update() {
        if (!this.isRunning) return;
        
        for (let step = 0; step < 200; step++) {
            this.solver.step();
            this.stepCount++;
        }
        
        this.draw();
        
        const maxPconc = this.solver.getMaxPconc();
        const maxSconc = this.solver.getMaxSconc();
        const totalMass = this.solver.calculateTotalMass();  // Add this
        
        document.getElementById('stepCount').textContent = this.stepCount;
        document.getElementById('maxPconc').textContent = maxPconc.toFixed(3);
        document.getElementById('maxSconc').textContent = maxSconc.toFixed(3);
        document.getElementById('totalMass').textContent = totalMass.toFixed(3);  // Add this

    }
    
    start() {
        if (this.animationId) return;
        this.isRunning = true;
        
        const fps = parseInt(document.getElementById('speed').value);
        const interval = 1000 / fps;
        
        const animate = () => {
            this.update();
            if (this.isRunning) {
                this.animationId = setTimeout(() => {
                    requestAnimationFrame(animate);
                }, interval);
            }
        };
        
        animate();
    }
    
    pause() {
        this.isRunning = false;
        if (this.animationId) {
            clearTimeout(this.animationId);
            this.animationId = null;
        }
    }
    
    reset() {
        const wasRunning = this.isRunning;
        this.pause();
        this.solver.reset();
        this.stepCount = 0;
        this.draw();
        document.getElementById('stepCount').textContent = '0';
        document.getElementById('maxPconc').textContent = this.solver.getMaxPconc().toFixed(3);
        document.getElementById('maxSconc').textContent = this.solver.getMaxSconc().toFixed(3);
        if (wasRunning) this.start();
    }

}

// Main application initialization
document.addEventListener('DOMContentLoaded', () => {
    const nx = 50;
    const ny = 50;
    const DP = document.getElementById('PdiffusionCoeff').value;
    const DS = document.getElementById('SdiffusionCoeff').value;
    const k = document.getElementById('kcoupling').value;
    // console.log('k', k);
    
    const solver = new DiffusionSolver(nx, ny, DP, DS, k);
    const visualizer = new DiffusionVisualizer('diffusionCanvas', solver, 500, 500);
    
    // Draw initial state
    visualizer.draw();
    
    // Setup event listeners
    document.getElementById('startBtn').addEventListener('click', () => visualizer.start());
    document.getElementById('pauseBtn').addEventListener('click', () => visualizer.pause());
    document.getElementById('resetBtn').addEventListener('click', () => visualizer.reset());
    
    
    const PDslider = document.getElementById('PdiffusionCoeff');
    const PDspan = document.getElementById('dPValue');
    // const PDupdate = () => PDspan.textContent = parseFloat(PDslider.value).toFixed(2);
    function PDupdate() {
        let value = parseFloat(PDslider.value);
        solver.DP = value;
        document.getElementById('dPValue').innerText = value.toFixed(2);
    }
    PDslider.addEventListener('input', PDupdate);

    const SDslider = document.getElementById('SdiffusionCoeff');
    const SDspan = document.getElementById('SPValue');
    function SDupdate() {
        let value = parseFloat(SDslider.value);
        solver.SP = value;
        document.getElementById('dSValue').innerText = value.toFixed(2);
    }
    SDslider.addEventListener('input', SDupdate);

    const kslider = document.getElementById('kcoupling');
    const kspan = document.getElementById('kValue');
    function kupdate() {
        let value = parseFloat(kslider.value);
        solver.k = value;
        document.getElementById('kValue').innerText = value.toFixed(2);
    }
    kslider.addEventListener('input', kupdate);

    const speedSlider = document.getElementById('speed');
    const speedValue = document.getElementById('speedValue');
    speedSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        speedValue.textContent = val;
        if (visualizer.isRunning) {
            // Restart animation with new speed
            visualizer.pause();
            visualizer.start();
        }
    });
});