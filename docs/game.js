// Browser port of flappy_env.py plus the trained PPO actor.
//
// The environment constants and the network weights both come from
// policy.json, which export_policy.py writes straight out of the trained
// model, so this file never hardcodes a number that Python also owns.
//
// Physics here must match flappy_env.py step for step: same fixed timestep,
// same update order (flap, then gravity, then pipes), same circle-vs-rectangle
// collision. Pipe layouts differ, because Python seeds a Mersenne Twister we
// do not reproduce -- but the policy is a function of the observation, not of
// the layout, so behaviour is preserved.

export function createNet(layers) {
  return function forward(input) {
    let x = input;
    for (const layer of layers) {
      const { W, b, act } = layer;
      const out = new Float64Array(b.length);
      for (let o = 0; o < b.length; o++) {
        const row = W[o];
        let sum = b[o];
        for (let i = 0; i < row.length; i++) sum += row[i] * x[i];
        out[o] = act === 'tanh' ? Math.tanh(sum) : sum;
      }
      x = out;
    }
    return x;
  };
}

// mulberry32: small seeded PRNG so runs are reproducible in the browser.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class FlappyGame {
  constructor(env, seed = 1) {
    this.env = env;
    this.reset(seed);
  }

  reset(seed) {
    const E = this.env;
    if (seed !== undefined) this.rand = mulberry32(seed);
    this.birdY = E.HEIGHT / 2;
    this.birdV = 0;
    this.score = 0;
    this.steps = 0;
    this.dead = false;

    const n = Math.floor(E.WIDTH / E.PIPE_SPACING) + 2;
    this.pipes = [];
    for (let i = 0; i < n; i++) {
      this.pipes.push({
        x: E.WIDTH + i * E.PIPE_SPACING,
        hole: E.HOLE_MIN + this.rand() * (E.HOLE_MAX - E.HOLE_MIN),
        scored: false,
      });
    }
  }

  gapTop(p) {
    return p.hole * this.env.HEIGHT;
  }

  gapBottom(p) {
    return (p.hole + this.env.HOLE_SIZE) * this.env.HEIGHT;
  }

  // Pipes still ahead of the bird, nearest first.
  upcoming() {
    const E = this.env;
    const ahead = this.pipes
      .filter((p) => p.x + E.PIPE_WIDTH >= E.BIRD_X - E.BIRD_RADIUS)
      .sort((a, b) => a.x - b.x);
    return ahead.length ? ahead : [...this.pipes].sort((a, b) => a.x - b.x);
  }

  observe() {
    const E = this.env;
    const ahead = this.upcoming();
    const pipe = ahead[0];
    const after = ahead.length > 1 ? ahead[1] : pipe;
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

    const dx = (pipe.x + E.PIPE_WIDTH - E.BIRD_X) / E.WIDTH;
    const dx2 = (after.x + E.PIPE_WIDTH - E.BIRD_X) / (E.WIDTH + E.PIPE_SPACING);
    const afterCentre = (this.gapTop(after) + this.gapBottom(after)) / 2;

    return [
      this.birdY / E.HEIGHT,
      this.birdV / E.MAX_FALL_SPEED,
      clamp(dx, 0, 1),
      (this.birdY - this.gapTop(pipe)) / E.HEIGHT,
      (this.birdY - this.gapBottom(pipe)) / E.HEIGHT,
      clamp(dx2, 0, 1),
      clamp((this.birdY - afterCentre) / E.HEIGHT, -1, 1),
    ];
  }

  // Circle-vs-rectangle, matching Pipe._hits_rect in flappy_env.py.
  hitsRect(top, bottom, px) {
    const E = this.env;
    const nx = Math.max(px, Math.min(E.BIRD_X, px + E.PIPE_WIDTH));
    const ny = Math.max(top, Math.min(this.birdY, bottom));
    const dx = E.BIRD_X - nx;
    const dy = this.birdY - ny;
    return dx * dx + dy * dy < E.BIRD_RADIUS * E.BIRD_RADIUS;
  }

  collides(p) {
    return (
      this.hitsRect(0, this.gapTop(p), p.x) ||
      this.hitsRect(this.gapBottom(p), this.env.HEIGHT, p.x)
    );
  }

  step(action) {
    const E = this.env;
    if (action === 1) this.birdV = E.JUMP_VELOCITY;
    this.birdV = Math.min(this.birdV + E.GRAVITY, E.MAX_FALL_SPEED);
    this.birdY += this.birdV;

    const rightmost = Math.max(...this.pipes.map((p) => p.x));
    for (const p of this.pipes) {
      p.x -= E.PIPE_SPEED;
      if (p.x + E.PIPE_WIDTH < 0) {
        p.x = rightmost + E.PIPE_SPACING;
        p.hole = E.HOLE_MIN + this.rand() * (E.HOLE_MAX - E.HOLE_MIN);
        p.scored = false;
      }
    }

    for (const p of this.pipes) {
      if (!p.scored && p.x + E.PIPE_WIDTH < E.BIRD_X - E.BIRD_RADIUS) {
        p.scored = true;
        this.score += 1;
      }
    }

    const hitGround = this.birdY + E.BIRD_RADIUS >= E.HEIGHT;
    const hitCeiling = this.birdY - E.BIRD_RADIUS <= 0;
    if (hitGround || hitCeiling || this.pipes.some((p) => this.collides(p))) {
      this.dead = true;
    }
    this.steps += 1;
    return this.dead;
  }
}
