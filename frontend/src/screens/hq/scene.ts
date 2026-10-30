/**
 * The Research HQ building: three floors of agents at their desks, and the data boxes they
 * carry to each other. Plain three.js, driven from React only through `setAgents`,
 * `handoff`, `select` and `dispose`.
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js'
import { buildPerson, ghost, type Person, type Pose } from './person'
import { type Agent, FLOOR_NAMES, type Floor, type Handoff } from './types'

const CANVAS = '#010102'
const SLAB = '#232429'
const EDGE = '#34343a'
const ACCENT = '#828fff'
const ACCENT_DEEP = '#5e6ad2'
const DESK = '#3a3b42'
const SCREEN_OFF = '#0f1011'
const SCREEN_ON = '#828fff'
const SCREEN_BLOCKED = '#e5484d'

const FLOOR_HEIGHT = 3.8
const SPACING = 3.4
const DEPTH = 7
/** The walkway in front of the desks, where couriers walk. */
const AISLE_Z = 1.6
const SEAT_Z = -0.9
/** Far enough sideways to clear the desk on the way out. */
const STEP_ASIDE = 1.05
/** In front of a desk, where a box is handed over. */
const HAND_Z = 0.85
const WALK_SPEED = 2.4
const LIFT_SPEED = 3.2
/** Couriers on the move at once; more wait in each agent's queue. */
const MAX_COURIERS = 6
/** Beyond this many waiting, a new handoff joins the last one instead of queueing. */
const MAX_QUEUE = 3

/** Shirt colours, picked by slot and floor so neighbours differ. */
const SHIRTS = [
  '#5e6ad2',
  '#3b82f6',
  '#27a644',
  '#f59e0b',
  '#d946ef',
  '#14b8a6',
  '#ef4444',
  '#a3a3a3',
  '#8b5cf6',
  '#0ea5e9',
  '#84cc16',
  '#f97316',
]

interface Seat {
  agent: Agent
  root: THREE.Group
  person: Person
  screen: THREE.MeshStandardMaterial
  ring: THREE.Mesh
  tag: CSS2DObject
  tagElement: HTMLDivElement
  /** Seconds left of the "just received something" glow. */
  flash: number
  /** True while the agent is out of its chair carrying a box. */
  away: boolean
  queue: Handoff[]
  extras: THREE.Object3D[]
}

interface Courier {
  handoff: Handoff
  box: THREE.Mesh
  label: CSS2DObject
  /** Path the box (and the walker, if any) follows, in world space. */
  path: THREE.Vector3[]
  /** Where the walker turns back: the index of the hand-over point in `path`. */
  handAt: number
  lengths: number[]
  travelled: number
  speed: number
  walker: Seat | null
  fade: number
}

export interface HqSceneOptions {
  onPick: (agentId: string | null) => void
  reducedMotion: boolean
}

export class HqScene {
  private readonly renderer: THREE.WebGLRenderer
  private readonly labels: CSS2DRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera: THREE.PerspectiveCamera
  private readonly controls: OrbitControls
  private readonly clock = new THREE.Clock()
  private readonly raycaster = new THREE.Raycaster()
  private readonly building = new THREE.Group()
  private readonly seats = new Map<string, Seat>()
  private readonly couriers: Courier[] = []
  private readonly resize: ResizeObserver
  private frame = 0
  private running = false
  private width = 0
  private slots = 0
  private selected: string | null = null
  private readonly disposables: { dispose: () => void }[] = []
  private lift: THREE.Mesh | null = null
  /** Handoffs not animated because every courier was busy. */
  dropped = 0

  private readonly host: HTMLElement
  private readonly options: HqSceneOptions

  constructor(host: HTMLElement, options: HqSceneOptions) {
    this.host = host
    this.options = options
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.setClearColor(CANVAS)
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    host.appendChild(this.renderer.domElement)

    this.labels = new CSS2DRenderer()
    this.labels.domElement.style.position = 'absolute'
    this.labels.domElement.style.inset = '0'
    this.labels.domElement.style.pointerEvents = 'none'
    host.appendChild(this.labels.domElement)

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200)
    this.camera.position.set(6, 11, 27)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.minDistance = 10
    this.controls.maxDistance = 46
    this.controls.minPolarAngle = 0.35
    this.controls.maxPolarAngle = 1.35
    this.controls.target.set(0, FLOOR_HEIGHT * 1.35, 0)

    this.scene.fog = new THREE.Fog(CANVAS, 40, 90)
    this.scene.add(new THREE.HemisphereLight('#d8dbff', '#1a1a22', 1.5))
    const sun = new THREE.DirectionalLight('#ffffff', 1.8)
    sun.position.set(10, 22, 14)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    sun.shadow.camera.left = -18
    sun.shadow.camera.right = 18
    sun.shadow.camera.top = 18
    sun.shadow.camera.bottom = -18
    this.scene.add(sun)
    this.scene.add(this.building)

    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(40, 48),
      new THREE.MeshStandardMaterial({ color: '#07070a', roughness: 1 }),
    )
    ground.rotation.x = -Math.PI / 2
    ground.position.y = -0.31
    ground.receiveShadow = true
    this.scene.add(ground)
    this.track(ground.geometry, ground.material)

    this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown)
    this.renderer.domElement.addEventListener('pointerup', this.onPointerUp)
    document.addEventListener('visibilitychange', this.onVisibility)
    this.resize = new ResizeObserver(() => this.fit())
    this.resize.observe(host)
    this.fit()
    this.start()
  }

  // -- public ---------------------------------------------------------------------------

  /** Place, update or remove agents to match the roster. */
  setAgents(agents: Agent[]): void {
    const slots = Math.max(3, ...agents.map((a) => a.slot + 1))
    if (slots !== this.slots) {
      this.slots = slots
      this.buildShell(slots)
      for (const seat of this.seats.values()) this.removeSeat(seat)
      this.seats.clear()
    }
    const wanted = new Set(agents.map((a) => a.id))
    for (const [id, seat] of this.seats) {
      if (!wanted.has(id)) {
        this.removeSeat(seat)
        this.seats.delete(id)
      }
    }
    for (const agent of agents) {
      const seat = this.seats.get(agent.id)
      if (seat && seat.agent.floor === agent.floor && seat.agent.slot === agent.slot) {
        seat.agent = agent
        this.paint(seat)
      } else {
        if (seat) this.removeSeat(seat)
        this.seats.set(agent.id, this.addSeat(agent))
      }
    }
  }

  /** Animate one handoff. Unknown agents are ignored. */
  handoff(handoff: Handoff): void {
    const from = this.seats.get(handoff.from)
    const to = this.seats.get(handoff.to)
    if (!from || !to || from === to) return
    if (from.queue.length >= MAX_QUEUE) {
      const last = from.queue.at(-1)
      if (last && last.to === handoff.to) last.count += handoff.count
      else this.dropped += 1
      return
    }
    from.queue.push({ ...handoff })
  }

  select(agentId: string | null): void {
    this.selected = agentId
    for (const seat of this.seats.values()) this.paint(seat)
  }

  dispose(): void {
    this.running = false
    cancelAnimationFrame(this.frame)
    this.resize.disconnect()
    document.removeEventListener('visibilitychange', this.onVisibility)
    this.renderer.domElement.removeEventListener('pointerdown', this.onPointerDown)
    this.renderer.domElement.removeEventListener('pointerup', this.onPointerUp)
    for (const courier of this.couriers) this.endCourier(courier)
    for (const seat of this.seats.values()) this.removeSeat(seat)
    this.clearShell()
    for (const item of this.disposables) item.dispose()
    this.controls.dispose()
    this.renderer.dispose()
    this.renderer.domElement.remove()
    this.labels.domElement.remove()
  }

  // -- the building -----------------------------------------------------------------------

  private floorY(floor: Floor | number): number {
    return (floor - 1) * FLOOR_HEIGHT
  }

  private seatX(slot: number): number {
    return (slot - (this.slots - 1) / 2) * SPACING
  }

  private liftX(): number {
    return this.width / 2 + 1.1
  }

  private clearShell(): void {
    for (const child of [...this.building.children]) {
      this.building.remove(child)
      child.traverse((node) => {
        if (node instanceof THREE.Mesh || node instanceof THREE.LineSegments) {
          node.geometry.dispose()
          const mats = Array.isArray(node.material) ? node.material : [node.material]
          for (const mat of mats) mat.dispose()
        }
        if (node instanceof CSS2DObject) node.element.remove()
      })
    }
    this.lift = null
  }

  private buildShell(slots: number): void {
    this.clearShell()
    this.width = slots * SPACING + 1.6
    for (const floor of [1, 2, 3] as const) {
      const y = this.floorY(floor)
      const slab = new THREE.Mesh(
        new THREE.BoxGeometry(this.width, 0.3, DEPTH),
        new THREE.MeshStandardMaterial({ color: SLAB, roughness: 0.9 }),
      )
      slab.position.y = y - 0.15
      slab.receiveShadow = true
      this.building.add(slab)
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(slab.geometry),
        new THREE.LineBasicMaterial({ color: EDGE }),
      )
      edges.position.copy(slab.position)
      this.building.add(edges)
      // A lavender strip along the slab's front edge, like a lit fascia.
      const strip = new THREE.Mesh(
        new THREE.BoxGeometry(this.width, 0.04, 0.04),
        new THREE.MeshBasicMaterial({ color: ACCENT_DEEP }),
      )
      strip.position.set(0, y - 0.02, DEPTH / 2)
      this.building.add(strip)
      // Back wall: smoked glass.
      const wall = new THREE.Mesh(
        new THREE.BoxGeometry(this.width, FLOOR_HEIGHT - 0.3, 0.08),
        new THREE.MeshStandardMaterial({
          color: '#1a1c2e',
          transparent: true,
          opacity: 0.35,
          roughness: 0.2,
          metalness: 0.4,
        }),
      )
      wall.position.set(0, y + (FLOOR_HEIGHT - 0.3) / 2, -DEPTH / 2 + 0.04)
      this.building.add(wall)

      // A ceiling light over each floor, so the people below read clearly.
      const lamp = new THREE.PointLight('#e6e8ff', 14, this.width, 1.6)
      lamp.position.set(0, y + FLOOR_HEIGHT - 0.6, 1)
      this.building.add(lamp)
      const panel = new THREE.Mesh(
        new THREE.BoxGeometry(this.width * 0.6, 0.03, 0.5),
        new THREE.MeshBasicMaterial({ color: '#c9ccff' }),
      )
      panel.position.set(0, y + FLOOR_HEIGHT - 0.33, 0.6)
      this.building.add(panel)

      const sign = document.createElement('div')
      sign.className =
        'rounded-xs border border-hairline bg-surface-1/80 px-1.5 py-0.5 font-mono text-caption tracking-wide text-ink-subtle uppercase'
      sign.textContent = `${floor} · ${FLOOR_NAMES[floor]}`
      const label = new CSS2DObject(sign)
      label.position.set(-this.width / 2 + 0.3, y + 0.35, DEPTH / 2)
      label.center.set(0, 0.5)
      this.building.add(label)
    }
    // Corner columns, so the slabs read as one building.
    for (const x of [-this.width / 2 + 0.15, this.width / 2 - 0.15]) {
      for (const z of [-DEPTH / 2 + 0.15, DEPTH / 2 - 0.15]) {
        const column = new THREE.Mesh(
          new THREE.BoxGeometry(0.22, FLOOR_HEIGHT * 3, 0.22),
          new THREE.MeshStandardMaterial({ color: '#2c2d33', roughness: 0.6 }),
        )
        column.position.set(x, (FLOOR_HEIGHT * 3) / 2 - 0.3, z)
        column.castShadow = true
        this.building.add(column)
      }
    }
    // Roof slab.
    const roof = new THREE.Mesh(
      new THREE.BoxGeometry(this.width, 0.2, DEPTH),
      new THREE.MeshStandardMaterial({
        color: SLAB,
        transparent: true,
        opacity: 0.25,
      }),
    )
    roof.position.y = this.floorY(4) - 0.1
    this.building.add(roof)

    // The data lift: a glowing shaft at the right side, between all three floors.
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.32, 0.32, FLOOR_HEIGHT * 3, 20, 1, true),
      new THREE.MeshBasicMaterial({
        color: ACCENT,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    )
    shaft.position.set(this.liftX(), (FLOOR_HEIGHT * 3) / 2 - 0.3, 0)
    this.building.add(shaft)
    this.lift = shaft
    const core = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.03, FLOOR_HEIGHT * 3, 6),
      new THREE.MeshBasicMaterial({ color: ACCENT }),
    )
    core.position.copy(shaft.position)
    this.building.add(core)
  }

  // -- agents ---------------------------------------------------------------------------

  private addSeat(agent: Agent): Seat {
    const root = new THREE.Group()
    const y = this.floorY(agent.floor)
    root.position.set(this.seatX(agent.slot), y, 0)
    root.userData['agentId'] = agent.id
    this.scene.add(root)

    const extras: THREE.Object3D[] = []
    const desk = new THREE.Mesh(
      new THREE.BoxGeometry(1.7, 0.08, 0.8),
      new THREE.MeshStandardMaterial({ color: DESK, roughness: 0.6 }),
    )
    desk.position.set(0, 0.74, SEAT_Z + 0.75)
    desk.castShadow = true
    desk.receiveShadow = true
    root.add(desk)
    extras.push(desk)
    for (const x of [-0.78, 0.78]) {
      const leg = new THREE.Mesh(
        new THREE.BoxGeometry(0.06, 0.74, 0.7),
        new THREE.MeshStandardMaterial({ color: '#2a2b2f' }),
      )
      leg.position.set(x, 0.37, SEAT_Z + 0.75)
      root.add(leg)
      extras.push(leg)
    }
    const monitor = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 0.5, 0.05),
      new THREE.MeshStandardMaterial({ color: '#0b0b0e', roughness: 0.4 }),
    )
    monitor.position.set(0, 1.08, SEAT_Z + 0.95)
    root.add(monitor)
    extras.push(monitor)
    const screen = new THREE.MeshStandardMaterial({
      color: SCREEN_OFF,
      emissive: SCREEN_OFF,
      emissiveIntensity: 0,
    })
    // The screen faces the agent and the camera both: a lit panel on each side.
    for (const z of [-0.03, 0.03]) {
      const panel = new THREE.Mesh(new THREE.PlaneGeometry(0.72, 0.42), screen)
      panel.position.set(0, 1.08, SEAT_Z + 0.95 + z)
      if (z < 0) panel.rotation.y = Math.PI
      root.add(panel)
      extras.push(panel)
    }

    if (agent.id === 'brain') {
      // The BRAIN gateway: a server rack with blinking lights behind the operator.
      const rack = new THREE.Mesh(
        new THREE.BoxGeometry(0.9, 2.2, 0.7),
        new THREE.MeshStandardMaterial({ color: '#111216', roughness: 0.5, metalness: 0.5 }),
      )
      rack.position.set(1.15, 1.1, -1.9)
      rack.castShadow = true
      root.add(rack)
      extras.push(rack)
      for (let i = 0; i < 9; i++) {
        const led = new THREE.Mesh(
          new THREE.BoxGeometry(0.7, 0.04, 0.02),
          new THREE.MeshBasicMaterial({ color: i % 3 === 0 ? '#27a644' : ACCENT }),
        )
        led.position.set(1.15, 0.3 + i * 0.22, -1.54)
        led.userData['blink'] = i * 0.37
        root.add(led)
        extras.push(led)
      }
    }

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.55, 0.68, 40),
      new THREE.MeshBasicMaterial({
        color: ACCENT,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    )
    ring.rotation.x = -Math.PI / 2
    ring.position.set(0, 0.02, SEAT_Z)
    root.add(ring)
    extras.push(ring)

    const shirt = SHIRTS[(agent.floor * 5 + agent.slot) % SHIRTS.length] ?? ACCENT
    const person = buildPerson(shirt, agent.floor * 7 + agent.slot)
    person.group.position.set(0, 0, SEAT_Z)
    person.group.rotation.y = 0
    root.add(person.group)

    const tagElement = document.createElement('div')
    const tag = new CSS2DObject(tagElement)
    tag.position.set(0, 2.25, SEAT_Z)
    root.add(tag)

    const seat: Seat = {
      agent,
      root,
      person,
      screen,
      ring,
      tag,
      tagElement,
      flash: 0,
      away: false,
      queue: [],
      extras,
    }
    this.paint(seat)
    return seat
  }

  private removeSeat(seat: Seat): void {
    this.scene.remove(seat.root)
    seat.person.dispose()
    seat.tagElement.remove()
    for (const object of seat.extras) {
      if (object instanceof THREE.Mesh) {
        object.geometry.dispose()
        ;(object.material as THREE.Material).dispose()
      }
    }
  }

  /** Name tag, ghosting and screen colour from the agent's current data. */
  private paint(seat: Seat): void {
    const { agent } = seat
    const planned = agent.status === 'planned'
    ghost(seat.person, planned)
    const dot =
      planned || agent.state === 'idle'
        ? 'bg-ink-tertiary'
        : agent.state === 'blocked'
          ? 'bg-pnl-negative'
          : 'bg-pnl-positive'
    const picked = this.selected === agent.id
    seat.tagElement.className = [
      'flex flex-col items-center gap-0.5 rounded-sm border px-2 py-1 text-center whitespace-nowrap backdrop-blur-sm',
      picked ? 'border-primary bg-surface-2/90' : 'border-hairline bg-surface-1/80',
      planned ? 'opacity-60' : '',
    ].join(' ')
    seat.tagElement.replaceChildren()
    const name = document.createElement('span')
    name.className = 'flex items-center gap-1.5 text-body-compact font-medium text-ink'
    const status = document.createElement('span')
    status.className = `size-1.5 rounded-full ${dot}`
    name.append(status, agent.name)
    const sub = document.createElement('span')
    sub.className = 'font-mono text-caption text-ink-subtle'
    sub.textContent = planned
      ? 'Joining soon'
      : agent.state === 'working'
        ? 'Working'
        : agent.state === 'blocked'
          ? 'Blocked'
          : 'Idle'
    seat.tagElement.append(name, sub)
    const ringMaterial = seat.ring.material as THREE.MeshBasicMaterial
    ringMaterial.opacity = picked ? 0.9 : 0
  }

  // -- couriers -------------------------------------------------------------------------

  private launch(from: Seat, handoff: Handoff): void {
    const to = this.seats.get(handoff.to)
    if (!to) return
    const start = from.root.position.clone()
    const end = to.root.position.clone()
    const sameFloor = from.agent.floor === to.agent.floor
    const reduced = this.options.reducedMotion
    let path: THREE.Vector3[]
    let handAt: number
    let walker: Seat | null = null
    if (reduced) {
      path = [end.clone().add(new THREE.Vector3(0, 1.0, SEAT_Z + 0.75))]
      handAt = 0
    } else if (sameFloor) {
      walker = from
      // Out from behind the desk on its open side, along the aisle, up to the other desk.
      const side = end.x >= start.x ? 1 : -1
      path = [
        new THREE.Vector3(start.x, start.y, SEAT_Z),
        new THREE.Vector3(start.x + side * STEP_ASIDE, start.y, SEAT_Z),
        new THREE.Vector3(start.x + side * STEP_ASIDE, start.y, AISLE_Z),
        new THREE.Vector3(end.x, end.y, AISLE_Z),
        new THREE.Vector3(end.x, end.y, HAND_Z),
      ]
      handAt = path.length - 1
      path.push(...path.slice(0, -1).reverse())
    } else {
      const lift = this.liftX()
      path = [
        new THREE.Vector3(start.x, start.y + 0.85, SEAT_Z + 0.75),
        new THREE.Vector3(start.x, start.y + 0.85, AISLE_Z),
        new THREE.Vector3(lift, start.y + 0.85, AISLE_Z),
        new THREE.Vector3(lift, start.y + 0.85, 0),
        new THREE.Vector3(lift, end.y + 0.85, 0),
        new THREE.Vector3(lift, end.y + 0.85, AISLE_Z),
        new THREE.Vector3(end.x, end.y + 0.85, AISLE_Z),
        new THREE.Vector3(end.x, end.y + 0.85, SEAT_Z + 0.75),
      ]
      handAt = path.length - 1
    }
    const lengths = [0]
    for (let i = 1; i < path.length; i++) {
      lengths.push(
        (lengths[i - 1] ?? 0) + (path[i] as THREE.Vector3).distanceTo(path[i - 1] as THREE.Vector3),
      )
    }

    const box = new THREE.Mesh(
      new THREE.BoxGeometry(0.26, 0.2, 0.26),
      new THREE.MeshStandardMaterial({
        color: ACCENT,
        emissive: ACCENT,
        emissiveIntensity: 1.4,
        transparent: true,
        opacity: 1,
      }),
    )
    box.position.copy(path[0] as THREE.Vector3)
    this.scene.add(box)
    const labelElement = document.createElement('div')
    labelElement.className =
      'rounded-xs bg-primary px-1 font-mono text-caption leading-tight text-white shadow'
    labelElement.textContent = handoff.count > 1 ? `×${handoff.count}` : handoff.kind
    const label = new CSS2DObject(labelElement)
    label.position.set(0, 0.3, 0)
    box.add(label)

    if (walker) walker.away = true
    this.couriers.push({
      handoff,
      box,
      label,
      path,
      handAt,
      lengths,
      travelled: 0,
      speed: sameFloor ? WALK_SPEED : LIFT_SPEED,
      walker,
      fade: reduced ? 1.2 : 0,
    })
  }

  private endCourier(courier: Courier): void {
    this.scene.remove(courier.box)
    courier.box.geometry.dispose()
    ;(courier.box.material as THREE.Material).dispose()
    courier.label.element.remove()
    if (courier.walker) {
      courier.walker.away = false
      courier.walker.person.group.position.set(0, 0, SEAT_Z)
      courier.walker.person.group.rotation.y = 0
    }
  }

  private advanceCourier(courier: Courier, dt: number): boolean {
    if (courier.fade > 0) {
      courier.fade -= dt
      const mat = courier.box.material as THREE.MeshStandardMaterial
      mat.opacity = Math.max(0, Math.min(1, courier.fade))
      if (courier.fade <= 0) {
        this.deliver(courier)
        return false
      }
      return true
    }
    const total = courier.lengths.at(-1) ?? 0
    const handDistance = courier.lengths[courier.handAt] ?? total
    const before = courier.travelled
    courier.travelled = Math.min(total, courier.travelled + courier.speed * dt)
    if (before < handDistance && courier.travelled >= handDistance) this.deliver(courier)

    let i = 1
    while (i < courier.lengths.length - 1 && (courier.lengths[i] ?? 0) < courier.travelled) i++
    const a = courier.path[i - 1] as THREE.Vector3
    const b = courier.path[i] as THREE.Vector3
    const span = (courier.lengths[i] ?? 0) - (courier.lengths[i - 1] ?? 0)
    const t = span > 0 ? (courier.travelled - (courier.lengths[i - 1] ?? 0)) / span : 1
    const point = a.clone().lerp(b, t)

    if (courier.walker) {
      const root = courier.walker.root.position
      const person = courier.walker.person.group
      person.position.set(point.x - root.x, 0, point.z - root.z)
      const direction = b.clone().sub(a)
      if (direction.lengthSq() > 1e-6) person.rotation.y = Math.atan2(direction.x, direction.z)
      // Carried in front at chest height until handed over, then gone.
      const carrying = courier.travelled < handDistance
      courier.box.visible = carrying
      if (carrying) {
        const forward = new THREE.Vector3(
          Math.sin(person.rotation.y),
          0,
          Math.cos(person.rotation.y),
        )
        courier.box.position.set(point.x, root.y + 1.05, point.z).addScaledVector(forward, 0.32)
      }
    } else {
      courier.box.position.copy(point)
      courier.box.rotation.y += dt * 2
    }
    return courier.travelled < total
  }

  /** The box reaches its target: light it up and let the next stage start. */
  private deliver(courier: Courier): void {
    const target = this.seats.get(courier.handoff.to)
    if (target) target.flash = 1.2
    courier.box.visible = false
    courier.label.element.style.display = 'none'
  }

  // -- loop -----------------------------------------------------------------------------

  private start(): void {
    if (this.running) return
    this.running = true
    this.clock.getDelta()
    const tick = () => {
      if (!this.running) return
      this.frame = requestAnimationFrame(tick)
      this.step(Math.min(this.clock.getDelta(), 0.1))
    }
    tick()
  }

  private stop(): void {
    this.running = false
    cancelAnimationFrame(this.frame)
  }

  private step(dt: number): void {
    const t = this.clock.elapsedTime
    const reduced = this.options.reducedMotion

    // Start queued couriers while there is room.
    for (const seat of this.seats.values()) {
      if (this.couriers.length >= MAX_COURIERS) break
      if (seat.away || seat.queue.length === 0) continue
      const next = seat.queue.shift()
      if (next) this.launch(seat, next)
    }
    for (let i = this.couriers.length - 1; i >= 0; i--) {
      const courier = this.couriers[i] as Courier
      if (!this.advanceCourier(courier, dt)) {
        this.endCourier(courier)
        this.couriers.splice(i, 1)
      }
    }

    let index = 0
    for (const seat of this.seats.values()) {
      index += 1
      const { agent } = seat
      const live = agent.status === 'live'
      const working = live && (agent.state === 'working' || seat.flash > 0)
      const pose: Pose = seat.away ? 'walking' : working ? 'working' : 'idle'
      seat.person.pose(pose, reduced ? 0 : t, index * 1.3)

      seat.flash = Math.max(0, seat.flash - dt)
      const color =
        live && agent.state === 'blocked' ? SCREEN_BLOCKED : working ? SCREEN_ON : SCREEN_OFF
      seat.screen.color.set(color)
      seat.screen.emissive.set(color)
      const flicker = reduced ? 1 : 0.85 + Math.sin(t * 23 + index) * 0.15
      seat.screen.emissiveIntensity = working
        ? 0.9 * flicker + seat.flash
        : live && agent.state === 'blocked'
          ? 0.8
          : 0
      const ring = seat.ring.material as THREE.MeshBasicMaterial
      if (this.selected !== agent.id) ring.opacity = Math.min(0.9, seat.flash)
      for (const object of seat.extras) {
        const blink = object.userData['blink']
        if (typeof blink === 'number' && object instanceof THREE.Mesh) {
          object.visible = !live || reduced || Math.sin(t * 6 + blink * 9) > -0.6
        }
      }
    }
    if (this.lift) {
      const mat = this.lift.material as THREE.MeshBasicMaterial
      mat.opacity = 0.1 + (reduced ? 0 : Math.sin(t * 2) * 0.03)
    }

    this.controls.update()
    this.renderer.render(this.scene, this.camera)
    this.labels.render(this.scene, this.camera)
  }

  // -- input and housekeeping -------------------------------------------------------------

  private fit(): void {
    const { clientWidth: width, clientHeight: height } = this.host
    if (width === 0 || height === 0) return
    this.renderer.setSize(width, height)
    this.labels.setSize(width, height)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
  }

  private down: { x: number; y: number } | null = null

  private readonly onPointerDown = (event: PointerEvent) => {
    this.down = { x: event.clientX, y: event.clientY }
  }

  /** A click, not the end of an orbit drag, picks the agent under the pointer. */
  private readonly onPointerUp = (event: PointerEvent) => {
    const down = this.down
    this.down = null
    if (!down || Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) return
    const rect = this.renderer.domElement.getBoundingClientRect()
    const pointer = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    )
    this.raycaster.setFromCamera(pointer, this.camera)
    const roots = [...this.seats.values()].map((s) => s.root)
    const hit = this.raycaster.intersectObjects(roots, true)[0]
    let node: THREE.Object3D | null = hit?.object ?? null
    while (node && typeof node.userData['agentId'] !== 'string') node = node.parent
    this.options.onPick((node?.userData['agentId'] as string | undefined) ?? null)
  }

  private readonly onVisibility = () => {
    if (document.hidden) this.stop()
    else this.start()
  }

  private track(...items: { dispose: () => void }[]): void {
    this.disposables.push(...items)
  }
}
