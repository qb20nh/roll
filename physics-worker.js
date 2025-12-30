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
const STATE_STRIDE = 4; // ballX, ballY, ballAngle, containerAngle

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
            inertia: 0.5 * BALL_MASS * (BALL_RADIUS * BALL_RADIUS)
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
        const gravitySubDt = GRAVITY * subDt;
        const rB = b.radius;
        const rC = c.radius;
        const maxDist = rC - rB;
        const maxDistSq = maxDist * maxDist;

        const invMass = 1 / b.mass;
        const invInertiaB = 1 / b.inertia;
        const invInertiaC = 1 / c.inertia;

        // Effective mass for tangential impulse (constant for this system)
        const invM = invMass;
        const invIb = (rB * rB) * invInertiaB;
        const invIw = (rC * rC) * invInertiaC;
        const effMass = 1 / (invM + invIb + invIw);

        for (let i = 0; i < SUB_STEPS; i++) {
            // Integration
            b.vy += gravitySubDt;
            b.x += b.vx * subDt;
            b.y += b.vy * subDt;
            b.angle += b.angularVelocity * subDt;
            c.angle += c.angularVelocity * subDt;

            // Collision
            const x = b.x;
            const y = b.y;
            const distSq = x * x + y * y;

            // Avoid sqrt unless we might be colliding.
            if (distSq >= maxDistSq) {
                const dist = Math.sqrt(distSq);
                if (dist <= 0) continue;

                const invDist = 1 / dist;
                const nx = x * invDist;
                const ny = y * invDist;

                // Fix penetration (only if needed)
                const pen = dist - maxDist;
                if (pen > 0) {
                    b.x -= nx * pen;
                    b.y -= ny * pen;
                }

                // Tangent
                const tx = -ny;
                const ty = nx;

                // Normal velocity
                const vn = b.vx * nx + b.vy * ny;

                if (vn > 0) {
                    // Normal impulse: j/m simplifies to -(1+e)*vn
                    const impulseN = -(1 + RESTITUTION_NORMAL) * vn;
                    b.vx += impulseN * nx;
                    b.vy += impulseN * ny;

                    // Tangential relative velocity at contact
                    const vtBallSurf = (b.vx * tx + b.vy * ty) + (b.angularVelocity * rB);
                    const vtWallSurf = c.angularVelocity * rC;
                    const vRelTan = vtBallSurf - vtWallSurf;

                    // Tangential impulse
                    const jt = -(1 + RESTITUTION_TANGENT) * vRelTan * effMass;

                    // Apply
                    const jtInvMass = jt * invMass;
                    b.vx += jtInvMass * tx;
                    b.vy += jtInvMass * ty;
                    b.angularVelocity += jt * rB * invInertiaB;
                    c.angularVelocity -= jt * rC * invInertiaC;
                }
            }
        }
        
        const stats = this.calculateEnergy();
        return this.correctEnergy(stats);
    }

    correctEnergy(stats) {
        if (!this.initialEnergy || this.initialEnergy < 0.000001) return stats.total;
        const currentKE = stats.total - stats.pe;
        const targetKE = this.initialEnergy - stats.pe;

        if (targetKE > 0.000001 && currentKE > 0.000001) {
            let scale = Math.sqrt(targetKE / currentKE);
            
            // Clamp scale to prevent instability from large corrections
            // Allows up to 1% adjustment per correction step
            scale = Math.max(0.99, Math.min(1.01, scale));
            
            // Safety check for NaN/Infinity
            if (!Number.isFinite(scale)) return stats.total;
            
            if (scale !== 1) {
                this.ball.vx *= scale;
                this.ball.vy *= scale;
                this.ball.angularVelocity *= scale;
                this.container.angularVelocity *= scale;
            }

            return stats.pe + currentKE * scale * scale;
        }

        return stats.total;
    }
}

// Worker state
let systems = [];
let systemIds = [];

// Message handler
self.onmessage = function(e) {
    const msg = e.data;
    const type = msg?.type;
    
    switch(type) {
        case 'init':
            // Initialize systems for this worker
            systemIds = msg.systemIds ?? msg.data?.systemIds ?? [];
            systems = systemIds.map((id) => new SimSystem(id));
            self.postMessage({ type: 'initialized' });
            break;
            
        case 'update':
            // Update all systems and write compact state to an output buffer.
            {
                const dt = msg.dt ?? msg.data?.dt;
                const batchId = msg.batchId;
                const expectedFloats = systems.length * STATE_STRIDE;
                let buffer = msg.buffer;
                if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== expectedFloats * 4) {
                    buffer = new ArrayBuffer(expectedFloats * 4);
                }

                const out = new Float32Array(buffer);
                let totalEnergy = 0;

                for (let i = 0; i < systems.length; i++) {
                    const s = systems[i];
                    totalEnergy += s.update(dt);

                    const base = i * STATE_STRIDE;
                    out[base] = s.ball.x;
                    out[base + 1] = s.ball.y;
                    out[base + 2] = s.ball.angle;
                    out[base + 3] = s.container.angle;
                }

                self.postMessage(
                    {
                        type: 'updated',
                        batchId,
                        buffer,
                        totalEnergy
                    },
                    [buffer]
                );
            }
            break;
            
        case 'reset':
            {
                const batchId = msg.batchId;
                const expectedFloats = systems.length * STATE_STRIDE;
                let buffer = msg.buffer;
                if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== expectedFloats * 4) {
                    buffer = new ArrayBuffer(expectedFloats * 4);
                }

                const out = new Float32Array(buffer);
                let totalEnergy = 0;

                for (let i = 0; i < systems.length; i++) {
                    const s = systems[i];
                    s.reset(s.id);
                    totalEnergy += s.initialEnergy;

                    const base = i * STATE_STRIDE;
                    out[base] = s.ball.x;
                    out[base + 1] = s.ball.y;
                    out[base + 2] = s.ball.angle;
                    out[base + 3] = s.container.angle;
                }

                self.postMessage(
                    {
                        type: 'reset',
                        batchId,
                        buffer,
                        totalEnergy
                    },
                    [buffer]
                );
            }
            break;
    }
};
