/** A stylised person built from primitives, posed each frame by the scene. */

import * as THREE from 'three'

export type Pose = 'idle' | 'working' | 'walking'

export interface Person {
  group: THREE.Group
  /** Every mesh, so the scene can raycast and grey out the whole figure. */
  meshes: THREE.Mesh[]
  pose: (pose: Pose, t: number, seed: number) => void
  dispose: () => void
}

const SKIN = ['#e8bfa0', '#c99572', '#8d5a3b', '#f1d3bd', '#a8714f']
const HAIR = ['#1d1612', '#3b2a1f', '#6b4a2e', '#c7a36a', '#2a2a30']
const TROUSERS = '#23252a'

function material(color: string): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.75, metalness: 0.05 })
}

/** `seed` picks skin and hair, so the same agent always looks the same. */
export function buildPerson(shirt: string, seed: number): Person {
  const group = new THREE.Group()
  const meshes: THREE.Mesh[] = []
  const geometries: THREE.BufferGeometry[] = []
  const materials: THREE.Material[] = []
  const add = (geometry: THREE.BufferGeometry, mat: THREE.Material, parent: THREE.Object3D) => {
    const mesh = new THREE.Mesh(geometry, mat)
    mesh.castShadow = true
    geometries.push(geometry)
    meshes.push(mesh)
    parent.add(mesh)
    return mesh
  }
  const skin = material(SKIN[seed % SKIN.length] ?? '#e8bfa0')
  const hair = material(HAIR[(seed * 3) % HAIR.length] ?? '#1d1612')
  const cloth = material(shirt)
  const legs = material(TROUSERS)
  materials.push(skin, hair, cloth, legs)

  // Legs hang from hip pivots, so a rotation swings them from the hip.
  const hipL = new THREE.Group()
  const hipR = new THREE.Group()
  hipL.position.set(-0.11, 0.78, 0)
  hipR.position.set(0.11, 0.78, 0)
  group.add(hipL, hipR)
  const legGeometry = new THREE.CapsuleGeometry(0.075, 0.55, 4, 8)
  add(legGeometry, legs, hipL).position.y = -0.36
  add(legGeometry.clone(), legs, hipR).position.y = -0.36

  const torso = new THREE.Group()
  torso.position.y = 0.78
  group.add(torso)
  add(new THREE.CapsuleGeometry(0.2, 0.42, 4, 12), cloth, torso).position.y = 0.33

  const shoulderL = new THREE.Group()
  const shoulderR = new THREE.Group()
  shoulderL.position.set(-0.27, 0.62, 0)
  shoulderR.position.set(0.27, 0.62, 0)
  torso.add(shoulderL, shoulderR)
  const armGeometry = new THREE.CapsuleGeometry(0.06, 0.42, 4, 8)
  add(armGeometry, cloth, shoulderL).position.y = -0.26
  add(armGeometry.clone(), cloth, shoulderR).position.y = -0.26
  const handGeometry = new THREE.SphereGeometry(0.06, 10, 8)
  add(handGeometry, skin, shoulderL).position.y = -0.52
  add(handGeometry.clone(), skin, shoulderR).position.y = -0.52

  const head = new THREE.Group()
  head.position.y = 0.92
  torso.add(head)
  add(new THREE.SphereGeometry(0.17, 20, 16), skin, head).position.y = 0.12
  const cap = add(
    new THREE.SphereGeometry(0.18, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.55),
    hair,
    head,
  )
  cap.position.set(0, 0.14, -0.01)
  cap.rotation.x = -0.25
  const eyeMaterial = material('#0f1011')
  materials.push(eyeMaterial)
  const eyeGeometry = new THREE.SphereGeometry(0.022, 8, 6)
  add(eyeGeometry, eyeMaterial, head).position.set(-0.06, 0.14, 0.155)
  add(eyeGeometry.clone(), eyeMaterial, head).position.set(0.06, 0.14, 0.155)

  const pose = (kind: Pose, t: number, s: number) => {
    const phase = t + s
    if (kind === 'walking') {
      const swing = Math.sin(phase * 9) * 0.6
      hipL.rotation.x = swing
      hipR.rotation.x = -swing
      shoulderL.rotation.x = -swing * 0.7
      shoulderR.rotation.x = swing * 0.7
      torso.position.y = 0.78 + Math.abs(Math.cos(phase * 9)) * 0.04
      head.rotation.y = 0
      return
    }
    hipL.rotation.x = 0
    hipR.rotation.x = 0
    torso.position.y = 0.78 + Math.sin(phase * 1.6) * 0.008
    if (kind === 'working') {
      // Forearms forward over the keyboard, fingers tapping.
      shoulderL.rotation.x = -1.1 + Math.sin(phase * 18) * 0.08
      shoulderR.rotation.x = -1.1 + Math.sin(phase * 18 + 1.7) * 0.08
      head.rotation.y = Math.sin(phase * 0.7) * 0.08
      head.rotation.x = 0.12
    } else {
      shoulderL.rotation.x = Math.sin(phase * 1.6) * 0.04
      shoulderR.rotation.x = -Math.sin(phase * 1.6) * 0.04
      head.rotation.y = Math.sin(phase * 0.45) * 0.35
      head.rotation.x = 0
    }
  }

  return {
    group,
    meshes,
    pose,
    dispose: () => {
      for (const geometry of geometries) geometry.dispose()
      for (const mat of materials) mat.dispose()
    },
  }
}

/** Grey the figure out and make it see-through: an agent that is not built yet. */
export function ghost(person: Person, on: boolean): void {
  for (const mesh of person.meshes) {
    const mat = mesh.material as THREE.MeshStandardMaterial
    mat.transparent = on
    mat.opacity = on ? 0.28 : 1
    mat.depthWrite = !on
  }
}
