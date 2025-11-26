// Physics Worker for Parallel Simulation
// This worker handles physics updates for a subset of simulation systems

// --- Physics Constants ---
const SUB_STEPS = 16; 
const GRAVITY = 9.81 * 100;
const CONTAINER_RADIUS = 300;
const BALL_RADIUS = 30;
const BALL_MASS = 10;
const CONTAINER_MASS = 200;
const RESTITUTION_NORMAL = 1.0; 
const RESTITUTION_TANGENT = 1.0; 
const NUM_SYSTEMS = 64;

// --- Simulation Class ---
class SimSystem {
    constructor(id) {
        this.id = id;
        this.ball = {};
        this.container = {};
        this.initialEnergy = 0;
        this.reset(id);
    }

    reset(id) {
        const offset = (id / NUM_SYSTEMS * 0.02) - 0.01;
        
        this.ball = {
            x: 1 + offset, 
            y: -220,
            vx: 0, vy: 0,
            angle: 0, angularVelocity: 0,
            radius: BALL_RADIUS, mass: BALL_MASS, 
            inertia: 0.5 * BALL_MASS * (BALL_RADIUS * BALL_RADIUS),
            history: []
        };

        this.container = {
            angle: 0, angularVelocity: 0,
            radius: CONTAINER_RADIUS, mass: CONTAINER_MASS,
            inertia: CONTAINER_MASS * (CONTAINER_RADIUS * CONTAINER_RADIUS)
        };

        this.initialEnergy = this.calculateEnergy().total;
    }

    calculateEnergy() {
        const b = this.ball;
        const c = this.container;

        const v2 = b.vx * b.vx + b.vy * b.vy;
        const keLin = 0.5 * b.mass * v2;
        const keRotB = 0.5 * b.inertia * (b.angularVelocity * b.angularVelocity);
        const keRotC = 0.5 * c.inertia * (c.angularVelocity * c.angularVelocity);
        const pe = -b.mass * GRAVITY * b.y;

        return {
            total: keLin + keRotB + keRotC + pe,
            pe: pe
        };
    }

    update(dt) {
        const subDt = dt / SUB_STEPS;
        const b = this.ball;
        const c = this.container;

        for (let i = 0; i < SUB_STEPS; i++) {
            // Integration
            b.vy += GRAVITY * subDt;
            b.x += b.vx * subDt;
            b.y += b.vy * subDt;
            b.angle += b.angularVelocity * subDt;
            c.angle += c.angularVelocity * subDt;

            // Collision
            const distSq = b.x*b.x + b.y*b.y;
            const dist = Math.sqrt(distSq);
            const maxDist = c.radius - b.radius;

            if (dist >= maxDist) {
                const nx = b.x / dist;
                const ny = b.y / dist;

                // Fix Penetration
                const pen = dist - maxDist;
                b.x -= nx * pen;
                b.y -= ny * pen;

                // Tangent
                const tx = -ny;
                const ty = nx;

                // Velocities
                const vn = b.vx * nx + b.vy * ny;
                
                const vt_ball_surf = (b.vx * tx + b.vy * ty) + b.angularVelocity * b.radius;
                const vt_wall_surf = c.angularVelocity * c.radius;
                const v_rel_tan = vt_ball_surf - vt_wall_surf;

                if (vn > 0) {
                    // Normal Impulse
                    const jn = -(1 + RESTITUTION_NORMAL) * vn * b.mass;
                    b.vx += (jn * nx) / b.mass;
                    b.vy += (jn * ny) / b.mass;

                    // Tangent Impulse
                    const invM = 1 / b.mass;
                    const invIb = (b.radius * b.radius) / b.inertia;
                    const invIw = (c.radius * c.radius) / c.inertia;
                    const effMass = 1 / (invM + invIb + invIw);

                    const jt = -(1 + RESTITUTION_TANGENT) * v_rel_tan * effMass;

                    // Apply
                    b.vx += (jt * tx) / b.mass;
                    b.vy += (jt * ty) / b.mass;
                    b.angularVelocity += (jt * b.radius) / b.inertia;
                    c.angularVelocity -= (jt * c.radius) / c.inertia;
                }
            }
        }
        
        this.correctEnergy();
    }

    correctEnergy() {
        if (!this.initialEnergy || this.initialEnergy < 0.000001) return;
        const stats = this.calculateEnergy();
        const currentKE = stats.total - stats.pe;
        const targetKE = this.initialEnergy - stats.pe;

        if (targetKE > 0.000001 && currentKE > 0.000001) {
            const reqScale = Math.sqrt(targetKE / currentKE);
            const smoothScale = 1 + (reqScale - 1) * 0.1;
            
            this.ball.vx *= smoothScale;
            this.ball.vy *= smoothScale;
            this.ball.angularVelocity *= smoothScale;
            this.container.angularVelocity *= smoothScale;
        }
    }

    // Serialize for message passing
    serialize() {
        return {
            id: this.id,
            ball: {
                x: this.ball.x,
                y: this.ball.y,
                angle: this.ball.angle,
                vx: this.ball.vx,
                vy: this.ball.vy,
                angularVelocity: this.ball.angularVelocity
            },
            container: {
                angle: this.container.angle,
                angularVelocity: this.container.angularVelocity
            },
            initialEnergy: this.initialEnergy
        };
    }
}

// Worker state
let systems = [];

// Message handler
self.onmessage = function(e) {
    const { type, data } = e.data;
    
    switch(type) {
        case 'init':
            // Initialize systems for this worker
            systems = data.systemIds.map(id => new SimSystem(id));
            self.postMessage({ type: 'initialized' });
            break;
            
        case 'update':
            // Update all systems
            const dt = data.dt;
            systems.forEach(s => s.update(dt));
            
            // Send back state
            const states = systems.map(s => s.serialize());
            const totalEnergy = systems.reduce((sum, s) => sum + s.calculateEnergy().total, 0);
            
            self.postMessage({ 
                type: 'updated',
                states: states,
                totalEnergy: totalEnergy
            });
            break;
            
        case 'reset':
            systems.forEach(s => s.reset(s.id));
            const resetStates = systems.map(s => s.serialize());
            const resetEnergy = systems.reduce((sum, s) => sum + s.calculateEnergy().total, 0);
            self.postMessage({ 
                type: 'reset',
                states: resetStates,
                totalEnergy: resetEnergy
            });
            break;
    }
};
