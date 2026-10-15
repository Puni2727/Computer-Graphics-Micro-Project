# =============================================================================
#  PAINTER'S ALGORITHM  -  LIVE 3D INDOOR NAVIGATION PROTOTYPE  (Blender 4.x / 5.x)
# =============================================================================
#  Run:  Blender > Scripting workspace > open this file > Run Script.
#  The script builds the furnished building, switches to the Layout workspace,
#  maximises the 3D viewport and starts the first-person walkthrough by itself.
#  (If auto-start fails: hover the 3D viewport, press F3, search
#   "Start Painter Walkthrough".)
#
#  WHAT YOU SEE
#  * The normal Blender viewport (which uses a Z-buffer) - the "reference".
#  * A PAINTER'S CANVAS overlay. It is a small software renderer written in
#    this file that draws every triangle of the scene with the depth test
#    DISABLED, using only the Painter's Algorithm:
#        1. DEPTH   d = (P - Camera) . forward  for every triangle
#        2. SORT    farthest -> nearest
#        3. PAINT   one triangle at a time, far first
#        4. OVERDRAW near triangles overwrite far ones
#    plus back-face culling, near-plane clipping, depth ruler and a
#    "wrong order" switch that shows why the sort order matters.
#
#  CONTROLS
#    UP / DOWN      walk forward / backward (W / S also work)
#    LEFT / RIGHT   turn left / right        A / D strafe      Shift sprint
#    Q / E          eye height down / up     PgUp / PgDn  look up / down
#    M              toggle mouse look (off by default)
#    Doors open by themselves when you come within ~3.8 m of a doorway.
#    F              autopilot tour along the navigation path (through rooms)
#    SPACE          pause / resume the painting sweep
#    R              restart sweep from the farthest triangle
#    . / ,          step painting forward / back (Shift = single triangle)
#    + / -          sweep faster / slower
#    I              instant mode (finished frame every frame, like a renderer)
#    V              WRONG ORDER (near -> far): near objects get covered!
#    X              sort key: triangle centroid  <->  farthest vertex
#    B              back-face culling on / off
#    C              colours: shaded / depth heat-map / paint order
#    L              show triangle edges
#    T              canvas: picture-in-picture / full screen / hidden
#    P              print far-to-near depth table to the System Console
#    H              show / hide help          Esc   exit walkthrough
# =============================================================================

import bpy
import math
import time
import random
import colorsys
import numpy as np
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
from math import radians

NS = bpy.app.driver_namespace

# Performance switches (smooth walking matters more than eye-candy):
USE_BEVEL = False       # True = rounded furniture edges (prettier renders, slower viewport)


# =============================================================================
#  0. CLEAN UP A PREVIOUS RUN
# =============================================================================
def _cleanup_previous():
    old = NS.get("painter_state")
    if old is not None:
        old.quit = True
    h = NS.get("painter_draw_handle")
    if h is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except Exception:
            pass
    NS["painter_draw_handle"] = None
    fn = NS.get("painter_timer_fn")
    if fn is not None:
        try:
            bpy.app.timers.unregister(fn)
        except Exception:
            pass
    NS["painter_timer_fn"] = None
    fh = NS.get("painter_frame_handler")
    if fh is not None and fh in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(fh)
    NS["painter_frame_handler"] = None
    cls = NS.get("painter_op_cls")
    if cls is not None:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    NS["painter_op_cls"] = None


_cleanup_previous()

for _ob in list(bpy.data.objects):
    bpy.data.objects.remove(_ob, do_unlink=True)
for _blocks in (bpy.data.curves, bpy.data.meshes, bpy.data.cameras,
                bpy.data.lights, bpy.data.materials):
    for _b in list(_blocks):
        if _b.users == 0:
            _blocks.remove(_b)


# =============================================================================
#  1. PAINTER CORE  (pure numpy - no Blender calls)
# =============================================================================
# ---- PAINTER CORE BEGIN ----
NEAR = 0.12


def clip_near(Pc, parent, near=NEAR):
    """Clip camera-space triangles against the near plane (depth = -z >= near).
    Returns (triangles, parent_index). Triangles keep their parent's index so
    colour / normal can be looked up later."""
    d = -Pc[:, :, 2]
    front = d > near
    nf = front.sum(axis=1)
    out_p = [Pc[nf == 3]]
    out_i = [parent[nf == 3]]

    m1 = nf == 1
    if m1.any():
        P = Pc[m1]
        f = front[m1]
        par = parent[m1]
        i = f.argmax(axis=1)
        r = np.arange(len(P))
        A, B, C = P[r, i], P[r, (i + 1) % 3], P[r, (i + 2) % 3]
        dA, dB, dC = -A[:, 2], -B[:, 2], -C[:, 2]
        tB = ((near - dA) / (dB - dA))[:, None]
        tC = ((near - dA) / (dC - dA))[:, None]
        out_p.append(np.stack([A, A + (B - A) * tB, A + (C - A) * tC], axis=1))
        out_i.append(par)

    m2 = nf == 2
    if m2.any():
        P = Pc[m2]
        f = front[m2]
        par = parent[m2]
        i = (~f).argmax(axis=1)
        r = np.arange(len(P))
        Z, A, B = P[r, i], P[r, (i + 1) % 3], P[r, (i + 2) % 3]
        dZ, dA, dB = -Z[:, 2], -A[:, 2], -B[:, 2]
        tA = ((near - dZ) / (dA - dZ))[:, None]
        tB = ((near - dZ) / (dB - dZ))[:, None]
        A2 = Z + (A - Z) * tA
        B2 = Z + (B - Z) * tB
        out_p.append(np.stack([A, B, B2], axis=1))
        out_i.append(par)
        out_p.append(np.stack([A, B2, A2], axis=1))
        out_i.append(par)

    return np.concatenate(out_p), np.concatenate(out_i)


def painter_pipeline(P, Nrm, cam_loc, cam_rot, k, W, H,
                     cull=True, key_mode="CENTROID", wrong=False):
    """STEP 1-2 of the Painter's Algorithm.
    P    (T,3,3) world-space triangles     Nrm (T,3) world normals
    cam_rot (3,3) camera axes as columns   k = focal_length / sensor_width
    Returns screen triangles already sorted in PAINT ORDER (far -> near)."""
    if len(P) == 0:
        return None
    Pc = (P - cam_loc) @ cam_rot                   # camera space (x right, y up, -z fwd)
    depth = -Pc[:, :, 2]                           # STEP 1: depth along view direction
    keep = depth.max(axis=1) > NEAR
    if cull:                                       # back-face culling
        cen = P.mean(axis=1)
        keep &= ((Nrm * (cam_loc - cen)).sum(axis=1) > 0.0)
    after_cull = int(keep.sum())
    parents = np.nonzero(keep)[0]
    if len(parents) == 0:
        return None
    Pc2, par = clip_near(Pc[parents], parents)
    if len(Pc2) == 0:
        return None
    d = np.maximum(-Pc2[:, :, 2], 1e-4)
    sx = Pc2[:, :, 0] / d * (k * W) + W * 0.5      # perspective projection
    sy = Pc2[:, :, 1] / d * (k * W) + H * 0.5
    vis = ~((sx.max(1) < 0) | (sx.min(1) > W) | (sy.max(1) < 0) | (sy.min(1) > H))
    sx, sy, d, par = sx[vis], sy[vis], d[vis], par[vis]
    if len(par) == 0:
        return None
    key = d.max(axis=1) if key_mode == "FARTHEST" else d.mean(axis=1)
    order = np.argsort(-key, kind="stable")        # STEP 2: sort far -> near
    if wrong:
        order = order[::-1]                        # deliberately broken order
    scr = np.stack([sx, sy], axis=-1)[order]
    key = key[order]
    par = par[order]
    a = scr[:, 1] - scr[:, 0]
    b = scr[:, 2] - scr[:, 0]
    area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return {"scr": scr, "key": key, "par": par,
            "after_cull": after_cull, "clipped_extra": int(len(Pc2) - len(parents)),
            "area": float(area.sum())}


LIGHT_DIR = np.array([-0.35, 0.85, -0.40])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
BG = np.array([0.05, 0.07, 0.11])
NEAR_COL = np.array([0.12, 0.85, 0.48])
FAR_COL = np.array([0.12, 0.32, 0.95])


def colorize(res, Nrm, C, mode):
    par, key = res["par"], res["key"]
    n = len(par)
    lamb = np.clip((Nrm[par] * LIGHT_DIR).sum(axis=1), 0.0, 1.0)
    dmin, dmax = float(key.min()), float(key.max())
    if mode == 0:                                  # shaded true colours + depth fog
        fog = np.clip(key / 45.0, 0, 1)[:, None] * 0.35
        col = C[par] * (0.38 + 0.62 * lamb)[:, None] * (1 - fog) + BG * fog
    elif mode == 1:                                # depth heat-map (green near, blue far)
        s = ((key - dmin) / (dmax - dmin + 1e-6))[:, None]
        col = (NEAR_COL + (FAR_COL - NEAR_COL) * s) * (0.72 + 0.28 * lamb)[:, None]
    else:                                          # paint order (first painted -> last)
        t = np.linspace(0, 1, n)[:, None]
        col = 0.5 + 0.5 * np.cos(2 * np.pi * (t * 0.85 + np.array([0.0, 0.33, 0.67])))
        col = col * (0.7 + 0.3 * lamb)[:, None]
    return np.clip(col, 0, 1).astype(np.float32), dmin, dmax
# ---- PAINTER CORE END ----


# =============================================================================
#  2. MATERIALS
# =============================================================================
def mat(name, color, roughness=0.72, metallic=0.0, emit=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metallic
    if emit > 0:
        for key in ("Emission Color", "Emission"):
            if key in b.inputs:
                b.inputs[key].default_value = (*color, 1)
                break
        if "Emission Strength" in b.inputs:
            b.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        try:
            m.surface_render_method = 'BLENDED'
        except Exception:
            try:
                m.blend_method = 'BLEND'
            except Exception:
                pass
    return m


M = {
    "floor_corridor": mat("Floor | Corridor", (0.30, 0.34, 0.42), 0.35),
    "floor_reception": mat("Floor | Warm oak", (0.45, 0.27, 0.13), 0.4),
    "floor_office": mat("Floor | Office carpet", (0.16, 0.22, 0.30)),
    "floor_lab": mat("Floor | Kitchen tile", (0.62, 0.67, 0.70), 0.25),
    "floor_class": mat("Floor | Bedroom carpet", (0.45, 0.32, 0.38)),
    "wall_warm": mat("Wall | Warm plaster", (0.80, 0.73, 0.62)),
    "wall_cool": mat("Wall | Cool plaster", (0.66, 0.75, 0.80)),
    "wall_mid": mat("Wall | Corridor", (0.50, 0.54, 0.64)),
    "ceiling": mat("Ceiling", (0.90, 0.90, 0.88)),
    "ground": mat("Ground", (0.07, 0.09, 0.11), 0.9),
    "wood": mat("Furniture | Walnut", (0.27, 0.13, 0.06), 0.45),
    "desk": mat("Furniture | Oak", (0.58, 0.36, 0.17), 0.5),
    "blue": mat("Upholstery | Deep blue", (0.035, 0.12, 0.30)),
    "fabric": mat("Upholstery | Neutral", (0.62, 0.57, 0.49)),
    "glass": mat("Window glass", (0.30, 0.65, 0.85), 0.1, 0.1, alpha=0.5),
    "black": mat("Screen | Black", (0.008, 0.012, 0.018), 0.18, 0.1),
    "cream": mat("Bedding | Cream", (0.88, 0.82, 0.69)),
    "red": mat("Accent | Terracotta", (0.60, 0.14, 0.08)),
    "green": mat("Plant | Green", (0.07, 0.36, 0.12)),
    "metal": mat("Furniture | Metal", (0.30, 0.34, 0.40), 0.3, 0.6),
    "white": mat("Whiteboard", (0.90, 0.92, 0.95), 0.35),
    "door": mat("Doors | Amber", (0.52, 0.26, 0.09), 0.5),
    "route": mat("Navigation Path | Cyan", (0.02, 0.80, 1.0), 0.3, 0.1, emit=3.0),
    "arrow": mat("Direction Arrow | Yellow", (1.0, 0.68, 0.04), 0.35, emit=2.5),
    "text": mat("Text | Dark", (0.025, 0.04, 0.07)),
    "panel": mat("Depth Panel | Navy", (0.025, 0.055, 0.11)),
    "near": mat("Depth | Near", (0.12, 0.85, 0.48), emit=2.0),
    "orb": mat("Walker Orb", (0.2, 0.9, 1.0), 0.2, emit=8.0),
    "lamp": mat("Lamp glow", (1.0, 0.80, 0.52), 0.4, emit=6.0),
    "tv": mat("TV animated", (0.1, 0.4, 0.9), 0.2, emit=2.5),
    "monitor": mat("Monitor glow", (0.2, 0.6, 0.9), 0.2, emit=1.8),
}
BOOK_MATS = [mat(f"Book {i}", c) for i, c in enumerate([
    (0.55, 0.08, 0.08), (0.08, 0.25, 0.55), (0.10, 0.45, 0.20),
    (0.70, 0.55, 0.10), (0.40, 0.12, 0.45), (0.85, 0.85, 0.80)])]


def assign(obj, material):
    if material:
        obj.data.materials.append(material)
    return obj


# =============================================================================
#  3. GEOMETRY HELPERS
# =============================================================================
DOORS = []        # dict(hinge, side, cx, cz)
FAN_PIVOTS = []
DYN_PAIRS = []    # (pivot, child) meshes the painter must re-pose every frame
CEILINGS = []
ARROWS = []


def cube(name, loc, scale, material, bevel=0.04, skip=False):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign(o, material)
    if bevel and USE_BEVEL:
        mod = o.modifiers.new("Soft edges", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        o.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    if skip:
        o["painter_skip"] = True
    return o


def prim(kind, material, name, skip=False, **kw):
    if kind == 'cyl':
        bpy.ops.mesh.primitive_cylinder_add(**kw)
    elif kind == 'cone':
        bpy.ops.mesh.primitive_cone_add(**kw)
    elif kind == 'uv':
        bpy.ops.mesh.primitive_uv_sphere_add(**kw)
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(**kw)
    o = bpy.context.object
    o.name = name
    assign(o, material)
    if skip:
        o["painter_skip"] = True
    return o


def text_obj(body, loc, size=0.34, material=None, name=None, rotation=(0, 0, 0)):
    curve = bpy.data.curves.new(name or body, 'FONT')
    curve.body = body
    curve.size = size
    curve.align_x = 'CENTER'
    curve.extrude = 0.001
    ob = bpy.data.objects.new(name or body, curve)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = rotation
    if material:
        assign(ob, material)
    return ob


def wall(name, a, b, height=3.0, thickness=0.18, material=None):
    x1, z1 = a
    x2, z2 = b
    length = math.hypot(x2 - x1, z2 - z1)
    dims = (thickness, height, length) if abs(x2 - x1) < 0.001 else (length, height, thickness)
    return cube(name, ((x1 + x2) / 2, height / 2, (z1 + z2) / 2), dims,
                material or M["wall_warm"], 0.025)


def floor(name, x, z, w, d, material):
    return cube(name, (x, -0.12, z), (w, 0.24, d), material, 0.015)


def make_fan(name, x, z):
    pivot = bpy.data.objects.new(name + " | Hub", None)
    bpy.context.collection.objects.link(pivot)
    pivot.location = (x, 2.78, z)
    for i, (off, dims) in enumerate([((0.65, 0, 0), (1.3, 0.03, 0.2)), ((-0.65, 0, 0), (1.3, 0.03, 0.2)),
                                     ((0, 0, 0.65), (0.2, 0.03, 1.3)), ((0, 0, -0.65), (0.2, 0.03, 1.3))]):
        blade = cube(f"{name} Blade {i}", (0, 0, 0), dims, M["wood"], 0.0)
        blade.parent = pivot
        blade.location = off
        blade["painter_dynamic"] = True
        DYN_PAIRS.append((pivot, blade))
    FAN_PIVOTS.append(pivot)


# =============================================================================
#  4. BUILDING
# =============================================================================
# World axes: X across the building, Y = UP, Z along the cross direction.
floor("Corridor Floor", 10, 0, 20, 4, M["floor_corridor"])
floor("Living Room Floor", 5, 6, 10, 8, M["floor_reception"])
floor("Office Floor", 15, 6, 10, 8, M["floor_office"])
floor("Kitchen Floor", 5, -6, 10, 8, M["floor_lab"])
floor("Bedroom Floor", 15, -6, 10, 8, M["floor_class"])
cube("Ground", (10, -0.75, 0), (70, 0.9, 60), M["ground"], 0.0)

wall("West Outer Wall", (0, 2), (0, 10), material=M["wall_warm"])
wall("North Living Wall", (0, 10), (10, 10), material=M["wall_warm"])
wall("North Office Wall", (10, 10), (20, 10), material=M["wall_cool"])
wall("East Outer Wall", (20, 2), (20, 10), material=M["wall_cool"])
wall("West Kitchen Wall", (0, -10), (0, -2), material=M["wall_cool"])
wall("South Kitchen Wall", (0, -10), (10, -10), material=M["wall_cool"])
wall("South Bedroom Wall", (10, -10), (20, -10), material=M["wall_warm"])
wall("East Bedroom Wall", (20, -10), (20, -2), material=M["wall_warm"])
wall("Living Divider", (10, 2), (10, 10), material=M["wall_mid"])
wall("Kitchen Divider", (10, -10), (10, -2), material=M["wall_mid"])
wall("Corridor West End", (0, -2), (0, 2), material=M["wall_mid"])
wall("Corridor East End", (20, -2), (20, 2), material=M["wall_mid"])

for z, prefix in [(2, "North Corridor"), (-2, "South Corridor")]:
    wall(prefix + " left", (0, z), (4, z), material=M["wall_mid"])
    wall(prefix + " middle", (6, z), (14, z), material=M["wall_mid"])
    wall(prefix + " right", (16, z), (20, z), material=M["wall_mid"])
    cube(prefix + " Header A", (5, 2.6, z), (2, 0.8, 0.18), M["wall_mid"], 0.02)
    cube(prefix + " Header B", (15, 2.6, z), (2, 0.8, 0.18), M["wall_mid"], 0.02)

# Ceilings (shown only during the first-person walkthrough).
for nm, x, z, w, d in [("Corridor", 10, 0, 20, 4), ("Living Room", 5, 6, 10, 8),
                       ("Office", 15, 6, 10, 8), ("Kitchen", 5, -6, 10, 8),
                       ("Bedroom", 15, -6, 10, 8)]:
    c = cube(nm + " Ceiling", (x, 3.06, z), (w, 0.12, d), M["ceiling"], 0.0)
    c.hide_viewport = True
    c.hide_render = True
    CEILINGS.append(c)

# ---- LIVING ROOM
cube("Living Room Rug", (3.1, 0.015, 5.9), (5.2, 0.035, 4.0), M["fabric"], 0.02)
cube("Sofa Base", (2.0, 0.48, 5.1), (1.25, 0.55, 2.8), M["blue"])
cube("Sofa Back", (1.43, 0.92, 5.1), (0.22, 0.75, 2.8), M["blue"])
for i, z in enumerate((4.25, 5.1, 5.95), 1):
    cube(f"Sofa Seat Cushion {i}", (2.15, 0.79, z), (0.92, 0.15, 0.76), M["fabric"], 0.09)
    cube(f"Sofa Back Cushion {i}", (1.57, 1.10, z), (0.18, 0.55, 0.72), M["fabric"], 0.08)
cube("Coffee Table Top", (3.7, 0.48, 5.1), (1.9, 0.12, 1.15), M["desk"])
for x in (3.0, 4.4):
    for z in (4.7, 5.5):
        cube("Coffee Table Leg", (x, 0.24, z), (0.09, 0.42, 0.09), M["metal"], 0.015)
cube("TV Console", (5.5, 0.45, 9.15), (3.5, 0.7, 0.55), M["wood"])
cube("Television", (5.5, 1.65, 9.78), (3.1, 1.75, 0.12), M["black"], 0.035)
cube("TV Screen", (5.5, 1.66, 9.70), (2.92, 1.56, 0.025), M["tv"], 0.01)
cube("Side Table", (3.0, 0.4, 3.3), (0.65, 0.65, 0.65), M["wood"])
cube("Living Painting", (0.13, 1.9, 7.8), (0.05, 0.9, 1.5), M["red"], 0.01)
cube("Living Painting Frame", (0.10, 1.9, 7.8), (0.04, 1.0, 1.6), M["wood"], 0.01)

# ---- OFFICE
cube("Office Desk Top", (15, 0.82, 7.0), (3.2, 0.14, 1.55), M["desk"])
for x in (13.65, 16.35):
    for z in (6.4, 7.6):
        cube("Office Desk Leg", (x, 0.4, z), (0.10, 0.78, 0.10), M["metal"], 0.02)
cube("Monitor Stand", (15, 1.02, 7.45), (0.14, 0.28, 0.12), M["metal"])
cube("Office Monitor", (15, 1.42, 7.48), (0.95, 0.66, 0.12), M["black"], 0.04)
cube("Monitor Display", (15, 1.43, 7.405), (0.83, 0.52, 0.025), M["monitor"], 0.01)
cube("Keyboard", (15, 0.93, 6.7), (0.72, 0.035, 0.25), M["metal"], 0.015)
cube("Office Chair Seat", (15, 0.65, 5.65), (0.72, 0.16, 0.72), M["black"])
cube("Office Chair Back", (15, 1.12, 5.32), (0.72, 0.85, 0.16), M["black"])
cube("Office Chair Stem", (15, 0.35, 5.65), (0.12, 0.45, 0.12), M["metal"])
cube("Filing Cabinet", (19.0, 0.85, 4.2), (0.9, 1.7, 1.8), M["metal"])
for y in (0.45, 0.85, 1.25):
    cube("Cabinet Drawer Front", (18.53, y, 4.2), (0.035, 0.30, 1.55), M["white"], 0.015)
cube("Bookshelf", (10.5, 1.1, 7.5), (0.4, 2.2, 2.4), M["wood"], 0.02)
random.seed(7)
for row, y in enumerate((0.62, 1.12, 1.62)):
    for j in range(10):
        cube(f"Book {row}-{j}", (10.62, y, 6.5 + j * 0.2),
             (0.24, random.choice((0.30, 0.36, 0.40)), 0.12),
             random.choice(BOOK_MATS), 0.008)

# ---- KITCHEN
cube("Kitchen Counter", (3.4, 0.78, -9.1), (5.6, 0.16, 0.72), M["white"])
cube("Kitchen Base Cabinets", (3.4, 0.42, -9.1), (5.45, 0.68, 0.62), M["wood"])
cube("Kitchen Backsplash", (3.4, 1.25, -9.78), (5.55, 0.85, 0.08), M["white"], 0.015)
cube("Kitchen Sink", (2.1, 0.89, -9.08), (0.95, 0.06, 0.48), M["metal"], 0.04)
cube("Kitchen Hob", (4.25, 0.89, -9.08), (0.95, 0.07, 0.48), M["black"], 0.03)
for x in (4.0, 4.48):
    for z in (-9.23, -8.94):
        prim('cyl', M["metal"], "Hob Ring", vertices=24, radius=0.10, depth=0.035, location=(x, 0.94, z))
cube("Refrigerator", (8.5, 1.1, -8.7), (1.1, 2.2, 1.2), M["white"])
cube("Refrigerator Handle", (8.02, 1.15, -8.7), (0.035, 0.65, 0.05), M["metal"])
for x in (2.2, 4.3, 6.4):
    cube("Kitchen Upper Cabinet", (x, 2.0, -9.65), (1.7, 0.75, 0.48), M["wood"])
cube("Dining Table", (6.8, 0.72, -5.2), (2.5, 0.14, 1.3), M["desk"])
for x in (5.8, 7.8):
    for z in (-5.7, -4.7):
        cube("Dining Table Leg", (x, 0.35, z), (0.10, 0.65, 0.10), M["wood"])
for x, z in ((5.8, -6.15), (7.8, -6.15), (5.8, -4.25), (7.8, -4.25)):
    cube("Dining Chair Seat", (x, 0.48, z), (0.55, 0.12, 0.55), M["blue"])
    cube("Dining Chair Back", (x, 0.85, z + (0.22 if z < -5 else -0.22)), (0.55, 0.65, 0.12), M["blue"])

# ---- BEDROOM
cube("Bedroom Rug", (15.2, 0.015, -5.2), (5.4, 0.035, 4.0), M["fabric"], 0.02)
cube("Bed Frame", (15.0, 0.36, -6.3), (3.0, 0.48, 3.6), M["wood"])
cube("Bed Mattress", (15.0, 0.68, -6.3), (2.85, 0.22, 3.45), M["cream"], 0.09)
cube("Bed Duvet", (15.0, 0.83, -5.75), (2.72, 0.12, 2.25), M["fabric"], 0.08)
for x in (14.3, 15.7):
    cube("Bed Pillow", (x, 0.86, -7.55), (0.95, 0.16, 0.62), M["white"], 0.08)
for x in (12.8, 17.2):
    cube("Nightstand", (x, 0.48, -7.5), (0.75, 0.85, 0.75), M["wood"])
    cube("Bedside Lamp Base", (x, 0.96, -7.5), (0.25, 0.08, 0.25), M["metal"])
    prim('cone', M["lamp"], "Bedside Lamp Shade", vertices=24, radius1=0.24, radius2=0.14,
         depth=0.35, location=(x, 1.18, -7.5))
cube("Wardrobe", (19.0, 1.2, -4.0), (1.1, 2.4, 2.0), M["wood"])
for z in (-4.45, -3.55):
    cube("Wardrobe Door Panel", (18.42, 1.2, z), (0.04, 2.15, 0.82), M["desk"], 0.02)
cube("Bedroom Painting", (15.0, 1.9, -9.86), (2.0, 0.8, 0.05), M["blue"], 0.01)

# Plants
for idx, (x, z) in enumerate([(8.7, 9.0), (11.0, 9.0), (18.5, -9.0), (0.7, 1.3)], 1):
    prim('cyl', M["wood"], f"Plant Pot {idx}", vertices=20, radius=0.25, depth=0.45, location=(x, 0.23, z))
    prim('ico', M["green"], f"Plant Leaves {idx}", subdivisions=1, radius=0.48, location=(x, 0.95, z))

# Windows
for name, x, z, width in [("Living Window", 2.8, 9.88, 3.4), ("Office Window", 15.8, 9.88, 3.4),
                          ("Kitchen Window", 8.2, -9.88, 1.6), ("Bedroom Window", 18.0, -9.88, 1.8)]:
    cube(name + " Glass", (x, 1.85, z), (width, 1.15, 0.045), M["glass"], 0.01)
    for dx in (-width / 2, width / 2):
        cube(name + " Frame", (x + dx, 1.85, z - 0.04), (0.07, 1.28, 0.08), M["white"], 0.01)
    cube(name + " Sill", (x, 1.25, z - 0.05), (width + 0.15, 0.10, 0.12), M["white"], 0.01)

# Ceiling lights + real Blender lights
for name, x, z in [("Living", 5, 6), ("Office", 15, 6), ("Kitchen", 5, -6), ("Bedroom", 15, -6),
                   ("Corridor A", 3, 0), ("Corridor B", 10, 0), ("Corridor C", 17, 0)]:
    prim('uv', M["lamp"], name + " Ceiling Lamp", segments=24, ring_count=12, radius=0.22,
         location=(x, 2.86, z))
    bpy.ops.object.light_add(type='POINT', location=(x, 2.5, z))
    L = bpy.context.object
    L.name = name + " Light"
    L.data.energy = 220 if "Corridor" not in name else 120
    L.data.color = (1.0, 0.80, 0.55)
    L.data.shadow_soft_size = 1.2

make_fan("Living Fan", 7.6, 6.5)
make_fan("Bedroom Fan", 17.6, -5.4)

# ---- DOORS: hinge empty + panel. They open automatically when you approach.
for name, x, z, side in [("Living Door", 4, 2, -1), ("Office Door", 14, 2, -1),
                         ("Kitchen Door", 4, -2, 1), ("Bedroom Door", 14, -2, 1)]:
    hinge = bpy.data.objects.new(name + " | Hinge", None)
    bpy.context.collection.objects.link(hinge)
    hinge.location = (x, 0, z)
    panel = cube(name, (0, 0, 0), (2.0, 2.2, 0.12), M["door"], 0.025)
    panel.parent = hinge
    panel.location = (1, 1.1, 0)
    panel["painter_dynamic"] = True
    DYN_PAIRS.append((hinge, panel))
    DOORS.append({"hinge": hinge, "side": side, "cx": x + 1.0, "cz": z, "name": name,
                  "near": 99.0, "open": False})

# Room signs on the corridor side of each doorway
for label, x, z, flip in [("LIVING ROOM", 5, 1.8, True), ("OFFICE", 15, 1.8, True),
                          ("KITCHEN", 5, -1.8, False), ("BEDROOM", 15, -1.8, False)]:
    cube(label + " Sign", (x, 2.62, z), (2.5, 0.45, 0.12), M["panel"], 0.025)
    off = -0.075 if flip else 0.075
    text_obj(label, (x, 2.55, z + off), 0.22, M["white"], label + " Label",
             (0, radians(180) if flip else 0, 0))

# =============================================================================
#  5. NAVIGATION PATH, ARROWS, ANIMATED WALKER ORB
# =============================================================================
curve = bpy.data.curves.new("Navigation Path Curve", 'CURVE')
curve.dimensions = '3D'
curve.bevel_depth = 0.05
curve.bevel_resolution = 3
for pts in ([(1.5, 0), (18.5, 0)], [(5, 0), (5, 5.8)], [(15, 0), (15, 4.4)],
            [(5, 0), (5, -3.6)], [(15, 0), (15, -3.2)]):
    sp = curve.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for p, (x, z) in zip(sp.points, pts):
        p.co = (x, 0.035, z, 1)
path = bpy.data.objects.new("Navigation Path | Start to Destination", curve)
bpy.context.collection.objects.link(path)
assign(path, M["route"])
path["painter_skip"] = True

for i, x in enumerate((3.0, 4.5, 6.5, 8.0, 9.5, 11.0, 12.5, 14.0, 16.0, 17.5), 1):
    a = prim('cone', M["arrow"], f"Navigation Arrow {i}", vertices=3, radius1=0.22, radius2=0.0,
             depth=0.36, location=(x, 0.08, 0), rotation=(0, radians(90), 0))
    ARROWS.append(a)
for label, loc, material in [("START", (1.5, 0.25, -0.65), M["near"]), ("DESTINATION", (18.2, 0.25, -0.65), M["arrow"])]:
    prim('uv', material, label + " Marker", segments=20, ring_count=10, radius=0.2, location=loc)
    text_obj(label, (loc[0], 0.62, loc[2] - 0.15), 0.26, M["white"], label + " Text")

ORB = prim('uv', M["orb"], "Walker Orb", skip=True, segments=24, ring_count=12, radius=0.17,
           location=(1.5, 0.4, 0))
bpy.ops.object.light_add(type='POINT', location=(1.5, 0.6, 0))
_ol = bpy.context.object
_ol.name = "Walker Orb Light"
_ol.data.energy = 60
_ol.data.color = (0.2, 0.85, 1.0)
_ol.parent = ORB
_ol.location = (0, 0.25, 0)

# =============================================================================
#  6. CAMERAS, LIGHTS, WORLD, IN-SCENE HUD
# =============================================================================
scene = bpy.context.scene
bpy.ops.object.camera_add(location=(1.5, 1.65, 0.0))
WALKCAM = bpy.context.object
WALKCAM.name = "Walkthrough Camera"
WALKCAM.data.lens = 24
WALKCAM.data.sensor_width = 36
WALKCAM.data.sensor_fit = 'HORIZONTAL'
WALKCAM.data.clip_start = 0.08
WALKCAM.data.clip_end = 150

bpy.ops.object.camera_add(location=(10, 15.5, -23))
OVERCAM = bpy.context.object
OVERCAM.name = "Indoor Navigation Camera"
OVERCAM.data.lens = 34


def look_rotation(direction):
    """Euler rotation for a camera/light looking along `direction` with WORLD +Y as up.
    (Blender's to_track_quat('-Z','Y') would point the camera's top at world +Z, which
    rolls the picture by 90 degrees in this Y-up scene.)"""
    f = Vector(direction).normalized()
    up = Vector((0.0, 1.0, 0.0))
    right = f.cross(up)
    if right.length < 1e-6:                      # looking straight up/down: pick any right
        right = Vector((1.0, 0.0, 0.0))
    right.normalize()
    true_up = right.cross(f)
    return Matrix((right, true_up, -f)).transposed().to_euler()


def point_at(obj, target):
    obj.rotation_euler = look_rotation(Vector(target) - obj.location)


point_at(OVERCAM, (10, 0.8, 0))
scene.camera = OVERCAM

bpy.ops.object.light_add(type='AREA', location=(10, 15, -1))
key = bpy.context.object
key.name = "Overview Softbox"
key.data.energy = 3400
key.data.shape = 'DISK'
key.data.size = 18
point_at(key, (10, 0, 0))

# Render settings
engines = {i.identifier for i in scene.render.bl_rna.properties['engine'].enum_items}
scene.render.engine = ('BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else
                       'BLENDER_EEVEE' if 'BLENDER_EEVEE' in engines else 'CYCLES')
scene.render.resolution_x = 1600
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 240
scene.frame_set(1)
try:
    scene.view_settings.view_transform = 'AgX'
except Exception:
    pass
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.color = (0.05, 0.065, 0.09)
try:
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.04, 0.055, 0.085, 1)
    bg.inputs[1].default_value = 0.8
except Exception:
    pass

# Explanatory panel beside the building (visible in the overview camera / render)
cube("Painter HUD Panel", (22.5, 3.0, 0), (5.5, 6.2, 0.22), M["panel"], 0.08, skip=True)
for txt, y, sz, m in [("PAINTER'S ALGORITHM", 5.45, 0.42, "white"), ("1. Depth of every triangle", 4.65, 0.24, "white"),
                      ("2. Sort farthest -> nearest", 4.15, 0.24, "white"), ("3. Paint far triangles first", 3.65, 0.24, "white"),
                      ("4. Near triangles cover far", 3.15, 0.24, "white"),
                      ("GREEN = NEAR     BLUE = FAR", 2.45, 0.24, "route")]:
    text_obj(txt, (22.5, y, -0.14), sz, M[m], "HUD " + txt[:12], rotation=(0, radians(180), 0))

# =============================================================================
#  7. PAINTER STATE + GEOMETRY CACHE
# =============================================================================
class PainterState:
    def __init__(self):
        self.quit = False
        self.walking = False
        self.auto = False
        self.wrong = False
        self.cull = True
        self.key_mode = "CENTROID"
        self.color_mode = 0
        self.canvas_mode = 0           # 0 picture-in-picture, 1 full, 2 hidden
        self.edges = False
        self.help = True
        self.playing = True
        self.instant = False
        self.frac = 0.0
        self.hold = 0.0
        self.sweep_secs = 7.0
        self.area_ptr = None
        self.sP = self.sN = self.sC = self.sO = None
        self.names = []
        self.dyn = []
        self.cache_key = None
        self.res = None
        self.cols = None
        self.dmin = self.dmax = 0.0
        self.stats = {}
        self.total_all = 0
        self.eye = 1.65
        self.yaw = 0.0
        self.pitch = 0.0
        self.pos = [1.5, 0.0]
        self.maximized = False
        self.mouse_look = False
        self.ambient = False           # fans / TV / orb animation while walking (off = smooth)
        self.last_compute = 0.0


ST = PainterState()
NS["painter_state"] = ST


def _normals(tri):
    c = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(c, axis=1, keepdims=True)
    return c / np.maximum(ln, 1e-12)


def _obj_color(obj):
    m = obj.active_material
    if m is None and obj.material_slots:
        m = obj.material_slots[0].material
    return np.array(m.diffuse_color[:3] if m else (0.6, 0.6, 0.6), dtype=np.float64)


def _local_tris(obj):
    me = obj.data
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    if nt == 0:
        return None
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    ids = np.empty(nt * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", ids)
    return co.reshape(-1, 3).astype(np.float64)[ids.reshape(-1, 3)]


def build_static(st):
    """Collect every visible mesh triangle in world space (the 'scene database')."""
    bpy.context.view_layer.update()
    Ps, Cs, Os = [], [], []
    st.names = []
    st.dyn = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj.get("painter_skip") or obj.hide_viewport:
            continue
        try:
            if obj.hide_get():
                continue
        except Exception:
            pass
        loc_tris = _local_tris(obj)
        if loc_tris is None:
            continue
        st.names.append(obj.name)
        oi = len(st.names) - 1
        if obj.get("painter_dynamic"):
            piv = obj.parent
            st.dyn.append({"pivot": piv, "child": obj, "tris": loc_tris,
                           "color": _obj_color(obj), "oi": oi})
            continue
        Mw = np.array(obj.matrix_world, dtype=np.float64)
        tri = loc_tris @ Mw[:3, :3].T + Mw[:3, 3]
        Ps.append(tri)
        Cs.append(np.tile(_obj_color(obj), (len(tri), 1)))
        Os.append(np.full(len(tri), oi, dtype=np.int32))
    st.sP = np.concatenate(Ps)
    st.sN = _normals(st.sP)
    st.sC = np.concatenate(Cs)
    st.sO = np.concatenate(Os)
    st.cache_key = None


def dyn_world(st):
    Ps, Ns, Cs, Os = [st.sP], [st.sN], [st.sC], [st.sO]
    for d in st.dyn:
        piv, ch = d["pivot"], d["child"]
        Mw = np.array(Matrix.Translation(piv.location) @
                      Matrix.Rotation(piv.rotation_euler[1], 4, 'Y') @
                      Matrix.Translation(ch.location), dtype=np.float64)
        tri = d["tris"] @ Mw[:3, :3].T + Mw[:3, 3]
        Ps.append(tri)
        Ns.append(_normals(tri))
        Cs.append(np.tile(d["color"], (len(tri), 1)))
        Os.append(np.full(len(tri), d["oi"], dtype=np.int32))
    return np.concatenate(Ps), np.concatenate(Ns), np.concatenate(Cs), np.concatenate(Os)


def update_frame(st, cam, W, H):
    """Run STEP 1-2 for the current camera (cached until something moves)."""
    key = (tuple(np.round(cam.location, 3)), tuple(np.round(cam.rotation_euler, 3)), int(W), int(H),
           st.cull, st.key_mode, st.wrong, st.color_mode,
           tuple(round(d["pivot"].rotation_euler[1], 3) for d in st.dyn[:1] + st.dyn[-1:]),
           tuple(round(dd["hinge"].rotation_euler[1], 2) for dd in DOORS),
           round(FAN_PIVOTS[0].rotation_euler[1], 2) if FAN_PIVOTS else 0)
    if key == st.cache_key:
        return
    now = time.perf_counter()
    old = st.cache_key
    if (old is not None and st.res is not None and old[2:4] == key[2:4]
            and now - st.last_compute < 0.07):
        return          # throttle the canvas to ~14 updates/s so walking stays smooth
    st.last_compute = now
    st.cache_key = key
    P, Nrm, C, O = dyn_world(st)
    st.total_all = len(P)
    R = np.array(cam.rotation_euler.to_matrix(), dtype=np.float64)
    loc = np.array(cam.location, dtype=np.float64)
    k = cam.data.lens / cam.data.sensor_width
    res = painter_pipeline(P, Nrm, loc, R, k, W, H, st.cull, st.key_mode, st.wrong)
    if res is None:
        st.res = None
        return
    cols, dmin, dmax = colorize(res, Nrm, C, st.color_mode)
    res["obj"] = O[res["par"]]
    st.res, st.cols, st.dmin, st.dmax = res, cols, dmin, dmax
    area = res["area"]
    st.stats = {"total": st.total_all, "culled": st.total_all - res["after_cull"],
                "clipped": res["clipped_extra"], "visible": len(res["par"]),
                "overdraw": area / float(W * H)}


# =============================================================================
#  8. AMBIENT ANIMATION (timer for the viewport, frame handler for renders)
# =============================================================================
DOOR_RADIUS = 3.8      # metres: a door opens when you (or the orb) come this close
DOOR_OPEN_DEG = 85


def door_distance(d, x, z):
    """Distance from a point on the floor plan to the centre of a doorway."""
    return math.hypot(d["cx"] - x, d["cz"] - z)


def update_doors(dt, viewer=None):
    """Open each door smoothly when the viewer / walker orb is near, close it when far."""
    if viewer is None:
        viewer = (WALKCAM.location.x, WALKCAM.location.z)
    sources = [viewer, (ORB.location.x, ORB.location.z)]
    k = 1.0 - math.exp(-9.0 * max(dt, 0.001))
    for d in DOORS:
        near = min(door_distance(d, sx, sz) for sx, sz in sources)
        d["near"] = near
        target = radians(DOOR_OPEN_DEG * d["side"]) if near < DOOR_RADIUS else 0.0
        cur = d["hinge"].rotation_euler[1]
        new = cur + (target - cur) * k
        if abs(target - new) < 0.002:
            new = target
        d["hinge"].rotation_euler[1] = new
        d["open"] = abs(new) > radians(10)


def animate(t, dt):
    for p in FAN_PIVOTS:
        p.rotation_euler[1] = t * 9.0
    s = 0.5 - 0.5 * math.cos(t * 0.45)
    ORB.location = (1.5 + 17.0 * s, 0.4 + 0.05 * math.sin(t * 6), 0.0)
    for i, a in enumerate(ARROWS):
        sc = 1.0 + 0.3 * math.sin(t * 4.5 - i * 0.9)
        a.scale = (sc, sc, sc)
    r, g, b = colorsys.hsv_to_rgb((t * 0.07) % 1.0, 0.75, 1.0)
    try:
        bsdf = M["tv"].node_tree.nodes.get("Principled BSDF")
        for key in ("Emission Color", "Emission"):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (r, g, b, 1)
                break
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1)
    except Exception:
        pass
    update_doors(dt)
    if not ST.walking:
        ang = t * 0.18
        OVERCAM.location = (10 + 7 * math.sin(ang), 15.5 + 1.2 * math.sin(ang * 0.7), -23)
        point_at(OVERCAM, (10, 0.8, 0))


def _tag_redraw():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


_T0 = time.perf_counter()
_last = [time.perf_counter()]


def _ambient_timer():
    if NS.get("painter_timer_fn") is not _ambient_timer:
        return None
    try:
        if bpy.context.screen is not None and bpy.context.screen.is_animation_playing:
            return 0.1
    except Exception:
        pass
    if ST.walking and not ST.ambient:
        return 0.25         # walking: do NOT animate anything in the background (no shaking)
    now = time.perf_counter()
    dt = min(now - _last[0], 0.1)
    _last[0] = now
    animate(now - _T0, dt)
    _tag_redraw()
    return 1.0 / 30.0


def _on_frame(scn, *args):
    animate(scn.frame_current / 24.0, 1.0 / 24.0)


NS["painter_timer_fn"] = _ambient_timer
NS["painter_frame_handler"] = _on_frame
bpy.app.timers.register(_ambient_timer, first_interval=0.1)
bpy.app.handlers.frame_change_post.append(_on_frame)


# =============================================================================
#  9. DRAWING (gpu overlay with the DEPTH TEST OFF = pure painter's algorithm)
# =============================================================================
_SH = {}


def _shader(kind):
    if kind in _SH:
        return _SH[kind]
    names = {"u": ["UNIFORM_COLOR", "2D_UNIFORM_COLOR"],
             "c": ["FLAT_COLOR", "2D_FLAT_COLOR", "SMOOTH_COLOR", "2D_SMOOTH_COLOR"]}[kind]
    for n in names:
        try:
            _SH[kind] = gpu.shader.from_builtin(n)
            return _SH[kind]
        except Exception:
            continue
    raise RuntimeError("no builtin shader")


def _rect(x, y, w, h, col):
    sh = _shader("u")
    b = batch_for_shader(sh, 'TRI_FAN', {"pos": ((x, y), (x + w, y), (x + w, y + h), (x, y + h))})
    sh.bind()
    sh.uniform_float("color", col)
    b.draw(sh)


def _outline(pts, col, closed=True):
    sh = _shader("u")
    b = batch_for_shader(sh, 'LINE_LOOP' if closed else 'LINE_STRIP', {"pos": pts})
    sh.bind()
    sh.uniform_float("color", col)
    b.draw(sh)


def _text(x, y, s, size=13, col=(1, 1, 1, 1)):
    try:
        blf.size(0, size)
    except TypeError:
        blf.size(0, size, 72)
    blf.color(0, *col)
    blf.position(0, x, y, 0)
    blf.draw(0, s)


def draw_callback(st):
    if st.quit or not st.walking:
        return
    ctx = bpy.context
    if ctx.area is None or ctx.area.as_pointer() != st.area_ptr or ctx.region is None:
        return
    region = ctx.region
    rx, ry = region.width, region.height
    cam = WALKCAM
    sc = ctx.scene
    aspect = sc.render.resolution_x / sc.render.resolution_y
    if st.canvas_mode == 1:
        W = max(240, min(rx - 40, (ry - 70) * aspect))
        x0, y0 = (rx - W) / 2, 18
    else:
        W = max(240, min(560, rx * 0.46))
        x0 = rx - W - 20
    H = W / aspect
    if st.canvas_mode == 0:
        y0 = ry - H - 44
    W, H = int(W), int(H)

    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('ALPHA')
    res = None
    n_paint = 0
    if st.canvas_mode != 2 or st.help:
        update_frame(st, cam, W, H)
        res = st.res
    if res is not None:
        total = len(res["par"])
        n_paint = total if st.instant else min(total, int(math.ceil(st.frac * total)))
    else:
        total = 0

    # ---------------------------------------------------------------- canvas
    if st.canvas_mode != 2:
        gpu.state.blend_set('NONE')
        sh = _shader("c")
        top, bot = (0.03, 0.04, 0.07, 1), (0.08, 0.10, 0.16, 1)
        b = batch_for_shader(sh, 'TRI_FAN', {
            "pos": ((x0, y0), (x0 + W, y0), (x0 + W, y0 + H), (x0, y0 + H)),
            "color": (bot, bot, top, top)})
        b.draw(sh)
        try:
            gpu.state.scissor_test_set(True)
            gpu.state.scissor_set(int(x0), int(y0), W, H)
        except Exception:
            pass
        if res is not None and n_paint > 0:
            scr = res["scr"][:n_paint] + np.array([x0, y0])
            pos = np.ascontiguousarray(scr.reshape(-1, 2), dtype=np.float32)
            col = np.ones((n_paint * 3, 4), dtype=np.float32)
            col[:, :3] = np.repeat(st.cols[:n_paint], 3, axis=0)
            b = batch_for_shader(sh, 'TRIS', {"pos": pos, "color": col})
            b.draw(sh)
            gpu.state.blend_set('ALPHA')
            if st.edges and n_paint < 14000:
                e = np.stack([scr[:, 0], scr[:, 1], scr[:, 1], scr[:, 2], scr[:, 2], scr[:, 0]], axis=1)
                su = _shader("u")
                b = batch_for_shader(su, 'LINES', {"pos": np.ascontiguousarray(e.reshape(-1, 2), dtype=np.float32)})
                su.bind()
                su.uniform_float("color", (0, 0, 0, 0.35))
                b.draw(su)
            last = scr[n_paint - 1]
            try:
                gpu.state.line_width_set(3.0)
            except Exception:
                pass
            _outline([tuple(p) for p in last], (1.0, 0.85, 0.1, 1.0))
            try:
                gpu.state.line_width_set(1.0)
            except Exception:
                pass
        try:
            gpu.state.scissor_test_set(False)
        except Exception:
            pass
        gpu.state.blend_set('ALPHA')
        _outline(((x0, y0), (x0 + W, y0), (x0 + W, y0 + H), (x0, y0 + H)), (0.2, 0.85, 1.0, 1.0))
        title = "PAINTER'S CANVAS  -  depth test OFF, triangles drawn far -> near"
        if st.wrong:
            title = "PAINTER'S CANVAS  -  WRONG ORDER (near -> far): near objects get covered!"
        _text(x0, y0 + H + 8, title, 13, (1, 0.4, 0.35, 1) if st.wrong else (0.5, 0.95, 1, 1))

        # depth ruler (far -> near) with a marker at the depth being painted now
        bx, by, bw, bh = x0 + 14, y0 + 10, W - 28, 9
        segs = 40
        for i in range(segs):
            s = i / (segs - 1)
            c = FAR_COL + (NEAR_COL - FAR_COL) * s
            _rect(bx + bw * i / segs, by, bw / segs + 1, bh, (c[0], c[1], c[2], 0.95))
        _text(bx, by + bh + 4, "FAR", 11, (0.6, 0.75, 1, 1))
        _text(bx + bw - 28, by + bh + 4, "NEAR", 11, (0.5, 1, 0.7, 1))
        if res is not None and n_paint > 0:
            d = float(res["key"][n_paint - 1])
            s = (d - st.dmin) / (st.dmax - st.dmin + 1e-6)
            mx = bx + (1 - s) * bw
            _rect(mx - 2, by - 5, 4, bh + 10, (1, 0.9, 0.1, 1))

    # ------------------------------------------------------------------ HUD
    st_ = st.stats
    lines = []
    painting = res is not None and n_paint < total
    if res is not None:
        d = float(res["key"][n_paint - 1]) if n_paint > 0 else 0.0
        nm = st.names[int(res["obj"][n_paint - 1])] if n_paint > 0 else "-"
        lines = [
            (f"1 DEPTH    d = (P - Cam) . forward   {st_['total']} tris, {st_['culled']} back-faces culled, "
             f"+{st_['clipped']} from near-clip", (0.8, 0.9, 1, 1)),
            ("2 SORT     key = " + ("triangle centroid" if st.key_mode == "CENTROID" else "farthest vertex") +
             "    order = " + ("NEAR -> FAR  (WRONG)" if st.wrong else "FAR -> NEAR  (correct)"),
             (1, 0.45, 0.4, 1) if st.wrong else (0.8, 0.9, 1, 1)),
            (f"3 PAINT    {n_paint}/{total} triangles   d = {d:.1f} m   [{nm}]",
             (1, 0.9, 0.2, 1) if painting else (0.8, 0.9, 1, 1)),
            (f"4 OVERDRAW x{st_['overdraw']:.1f}   near triangles overwrite pixels painted by far ones",
             (0.45, 1, 0.6, 1) if not painting else (0.8, 0.9, 1, 1)),
        ]
    bx, by = 16, ry - 30
    boxw = 640 if st.help else 640
    rows = 2 + len(lines) + (9 if st.help else 1)
    _rect(bx - 8, by - rows * 19 + 8, boxw, rows * 19 + 12, (0.02, 0.03, 0.06, 0.78))
    y = by
    _text(bx, y, "PAINTER'S ALGORITHM  -  LIVE PROTOTYPE", 16, (0.3, 0.9, 1, 1))
    y -= 22
    for s, c in lines:
        _text(bx, y, s, 12, c)
        y -= 19
    flags = (f"cull:{'ON' if st.cull else 'OFF'}  colors:{['shaded', 'depth', 'order'][st.color_mode]}  "
             f"{'INSTANT' if st.instant else ('PLAYING' if st.playing else 'PAUSED')}  "
             f"{'AUTOPILOT' if st.auto else 'manual'}")
    _text(bx, y, flags, 11, (0.7, 0.75, 0.85, 1))
    y -= 19
    nd = min(DOORS, key=lambda dd: dd.get("near", 99.0))
    _text(bx, y, f"DOOR  nearest: {nd['name']}  {nd.get('near', 99.0):.1f} m  ->  "
                 f"{'OPEN' if nd.get('open') else 'closed'}   (opens within {DOOR_RADIUS:.1f} m)",
          11, (1, 0.8, 0.3, 1) if nd.get("open") else (0.7, 0.75, 0.85, 1))
    y -= 19
    if st.help:
        for s in ["UP/DOWN arrows walk | LEFT/RIGHT arrows turn | A/D strafe | Shift sprint",
                  "Q/E eye height | PgUp/PgDn look up/down | M mouse look | F autopilot tour",
                  "SPACE pause | R restart | . , step (Shift=1 tri) | + - speed | I instant",
                  "V wrong order | X sort key | B back-face cull | C colours | L edges",
                  "T canvas size | G solid/material view | N fans+TV animation | P table | H help | Esc",
                  "Blender's own viewport (left) uses a Z-buffer; the canvas uses ONLY sorting.",
                  "Known limit: cyclic / intersecting triangles cannot be ordered by depth alone."]:
            _text(bx, y, s, 11, (0.65, 0.72, 0.85, 1))
            y -= 19
    gpu.state.blend_set('NONE')


# =============================================================================
#  10. WALKTHROUGH OPERATOR
# =============================================================================
TOUR = [(1.5, 0.0, 0, 0), (5.0, 0.0, 0, -2), (5.0, 0.4, 90, -8), (5.0, 5.0, 90, -8),
        (5.0, 5.0, 180, -14), (5.0, 5.0, 90, -4), (5.0, 1.0, -90, -6), (5.0, -0.4, -90, -6),
        (5.0, -4.9, -90, -10), (5.0, -4.9, 0, -14), (5.0, -0.6, 0, -4),
        (15.0, 0.0, 0, -2), (15.0, 0.4, 90, -8), (15.0, 4.5, 90, -10), (15.0, 0.8, -90, -6),
        (15.0, -4.6, -90, -14), (15.0, -4.6, 180, -10), (15.0, -0.4, 0, -4), (18.4, 0.0, 0, 0)]

RECTS = [(0.45, 19.55, -1.55, 1.55), (0.45, 9.55, 2.45, 9.55), (10.45, 19.55, 2.45, 9.55),
         (0.45, 9.55, -9.55, -2.45), (10.45, 19.55, -9.55, -2.45),
         (4.25, 5.75, -2.5, 2.5), (14.25, 15.75, -2.5, 2.5)]


def allowed(x, z):
    return any(a <= x <= b and c <= z <= d for a, b, c, d in RECTS)


def angdiff(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def aim_camera(st):
    cam = WALKCAM
    d = Vector((math.cos(st.pitch) * math.cos(st.yaw), math.sin(st.pitch),
                math.cos(st.pitch) * math.sin(st.yaw)))
    cam.rotation_euler = look_rotation(d)
    cam.location = (st.pos[0], st.eye, st.pos[1])


class VIEW3D_OT_painter_walkthrough(bpy.types.Operator):
    """Walk through the building and watch the Painter's Algorithm paint it"""
    bl_idname = "view3d.painter_walkthrough"
    bl_label = "Start Painter Walkthrough"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        st = ST
        st.quit = False
        self.area = context.area
        if self.area is None or self.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run this from the 3D viewport")
            return {'CANCELLED'}
        st.area_ptr = self.area.as_pointer()
        self.space = self.area.spaces.active
        self.prev_overlay = getattr(self.space.overlay, "show_overlays", True)
        self.prev_gizmo = getattr(self.space, "show_gizmo", True)
        self.prev_shading = self.space.shading.type
        self.prev_persp = self.space.region_3d.view_perspective
        for c in CEILINGS:
            c.hide_viewport = False
        build_static(st)
        st.pos = [1.5, 0.0]
        st.yaw, st.pitch, st.eye = 0.0, 0.0, 1.65
        aim_camera(st)
        context.scene.camera = WALKCAM
        self.space.region_3d.view_perspective = 'CAMERA'
        try:
            self.space.shading.type = 'SOLID'          # fast, steady frame rate
            self.space.shading.color_type = 'MATERIAL'
            self.space.shading.show_cavity = False
        except Exception:
            pass
        try:
            self.space.overlay.show_overlays = False
            self.space.show_gizmo = False
        except Exception:
            pass
        st.walking = True
        st.frac = 0.0
        st.hold = 0.0
        self.keys = set()
        self.shift = False
        self.vel = Vector((0.0, 0.0))
        self.yaw_rate = 0.0
        self.last = time.perf_counter()
        self.leg_i = 0
        self.leg_from = None
        self.handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (st,), 'WINDOW', 'POST_PIXEL')
        NS["painter_draw_handle"] = self.handle
        self.timer = context.window_manager.event_timer_add(1.0 / 60.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        st.mouse_look = False      # arrow keys are the default; press M for mouse look
        self._warp(context)
        self.report({'INFO'}, "Arrow keys walk/turn | F autopilot | H help | Esc exit")
        return {'RUNNING_MODAL'}

    def _region(self):
        for r in self.area.regions:
            if r.type == 'WINDOW':
                return r
        return None

    def _warp(self, context):
        r = self._region()
        if r:
            self.cx = r.x + r.width // 2
            self.cy = r.y + r.height // 2
            try:
                context.window.cursor_warp(self.cx, self.cy)
            except Exception:
                pass

    def _finish(self, context):
        st = ST
        st.walking = False
        st.auto = False
        try:
            context.window_manager.event_timer_remove(self.timer)
        except Exception:
            pass
        try:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
        except Exception:
            pass
        NS["painter_draw_handle"] = None
        try:
            context.window.cursor_set('DEFAULT')
        except Exception:
            pass
        for c in CEILINGS:
            c.hide_viewport = True
        context.scene.camera = OVERCAM
        try:
            self.space.region_3d.view_perspective = self.prev_persp
            self.space.shading.type = self.prev_shading
            self.space.overlay.show_overlays = self.prev_overlay
            self.space.show_gizmo = self.prev_gizmo
        except Exception:
            pass
        self.area.tag_redraw()
        return {'FINISHED'}

    # --------------------------------------------------------------------
    def _begin_leg(self, from_pose):
        self.leg_from = from_pose
        to = TOUR[self.leg_i]
        dist = math.hypot(to[0] - from_pose[0], to[1] - from_pose[1])
        turn = abs(angdiff(from_pose[2], radians(to[2])))
        self.leg_dur = max(dist / 2.2, turn / radians(55), 0.7)
        self.leg_t = 0.0

    def _autopilot(self, dt):
        st = ST
        if self.leg_from is None:
            self._begin_leg((st.pos[0], st.pos[1], st.yaw, st.pitch))
        self.leg_t += dt
        s = min(1.0, self.leg_t / self.leg_dur)
        e = s * s * (3 - 2 * s)
        fx, fz, fy, fp = self.leg_from
        to = TOUR[self.leg_i]
        st.pos = [fx + (to[0] - fx) * e, fz + (to[1] - fz) * e]
        st.yaw = fy + angdiff(fy, radians(to[2])) * e
        st.pitch = fp + (radians(to[3]) - fp) * e
        if s >= 1.0:
            done = (to[0], to[1], radians(to[2]), radians(to[3]))
            self.leg_i = (self.leg_i + 1) % len(TOUR)
            self._begin_leg(done)

    def _tick(self, context):
        st = ST
        now = time.perf_counter()
        dt = min(now - self.last, 0.05)
        self.last = now
        if st.auto:
            self._autopilot(dt)
        else:
            fwd = Vector((math.cos(st.yaw), math.sin(st.yaw)))
            right = Vector((-math.sin(st.yaw), math.cos(st.yaw)))
            mv = Vector((0.0, 0.0))
            # UP / DOWN arrows (or W / S) walk forward / backward,
            # LEFT / RIGHT arrows turn, A / D strafe.
            if 'W' in self.keys or 'UP_ARROW' in self.keys: mv += fwd
            if 'S' in self.keys or 'DOWN_ARROW' in self.keys: mv -= fwd
            if 'D' in self.keys: mv += right
            if 'A' in self.keys: mv -= right
            if mv.length > 0:
                mv.normalize()
            target = mv * (5.0 if self.shift else 2.8)
            self.vel += (target - self.vel) * min(1.0, dt * 9.0)     # smooth accelerate / brake
            if self.vel.length < 0.01:
                self.vel = Vector((0.0, 0.0))
            if self.vel.length > 0:
                step = self.vel * dt
                nx, nz = st.pos[0] + step.x, st.pos[1] + step.y
                if allowed(nx, nz):
                    st.pos = [nx, nz]
                elif allowed(nx, st.pos[1]):
                    st.pos[0] = nx
                elif allowed(st.pos[0], nz):
                    st.pos[1] = nz
            if 'Q' in self.keys: st.eye = max(1.0, st.eye - 1.2 * dt)
            if 'E' in self.keys: st.eye = min(2.3, st.eye + 1.2 * dt)
            turn = (1.0 if 'RIGHT_ARROW' in self.keys else 0.0) - (1.0 if 'LEFT_ARROW' in self.keys else 0.0)
            self.yaw_rate += (turn * 1.7 - self.yaw_rate) * min(1.0, dt * 10.0)   # smooth turning
            if abs(self.yaw_rate) < 0.005:
                self.yaw_rate = 0.0
            st.yaw += self.yaw_rate * dt
            if 'PAGE_UP' in self.keys: st.pitch = min(1.2, st.pitch + 1.2 * dt)
            if 'PAGE_DOWN' in self.keys: st.pitch = max(-1.2, st.pitch - 1.2 * dt)
        aim_camera(st)
        update_doors(dt, (st.pos[0], st.pos[1]))   # doors react to where YOU are
        # painting sweep
        if st.instant:
            st.frac = 1.0
        elif st.playing:
            if st.frac >= 1.0:
                st.hold += dt
                if st.hold > 1.8:
                    st.frac, st.hold = 0.0, 0.0
            else:
                st.frac = min(1.0, st.frac + dt / st.sweep_secs)
        self.area.tag_redraw()

    def _print_table(self):
        st = ST
        if st.res is None:
            return
        r = st.res
        print("\n" + "=" * 78)
        print("PAINTER'S ALGORITHM - depth table (paint order: farthest first)")
        print("=" * 78)
        n = len(r["par"])
        for i in list(range(min(12, n))) + [None] + list(range(max(12, n - 12), n)):
            if i is None:
                print("   ...")
                continue
            print(f"{i + 1:05d}. depth = {r['key'][i]:7.2f} m | {st.names[int(r['obj'][i])]}")
        print("=" * 78 + "\n")

    def modal(self, context, event):
        st = ST
        if st.quit or not st.walking:
            return self._finish(context)
        if event.type == 'TIMER':
            self._tick(context)
            return {'RUNNING_MODAL'}
        self.shift = event.shift
        if event.type == 'ESC' and event.value == 'PRESS':
            return self._finish(context)
        if event.type == 'MOUSEMOVE':
            if st.mouse_look and not st.auto and hasattr(self, "cx"):
                dx, dy = event.mouse_x - self.cx, event.mouse_y - self.cy
                if dx or dy:
                    st.yaw += dx * 0.0022
                    st.pitch = max(-1.2, min(1.2, st.pitch + dy * 0.0022))
                    self._warp(context)
            return {'RUNNING_MODAL'}
        held = {'W', 'A', 'S', 'D', 'Q', 'E', 'LEFT_ARROW', 'RIGHT_ARROW', 'UP_ARROW', 'DOWN_ARROW',
                'PAGE_UP', 'PAGE_DOWN'}
        if event.type in held:
            if event.value == 'PRESS':
                self.keys.add(event.type)
                if st.auto and event.type in {'W', 'A', 'S', 'D', 'LEFT_ARROW', 'RIGHT_ARROW',
                                              'UP_ARROW', 'DOWN_ARROW'}:
                    st.auto = False          # any manual key takes over from the autopilot
            elif event.value == 'RELEASE':
                self.keys.discard(event.type)
            return {'RUNNING_MODAL'}
        if event.value != 'PRESS':
            return {'RUNNING_MODAL'}
        t = event.type
        total = len(st.res["par"]) if st.res is not None else 1
        if t == 'SPACE':
            st.playing = not st.playing
            st.instant = False
        elif t == 'R':
            st.frac, st.hold, st.playing, st.instant = 0.0, 0.0, True, False
        elif t in {'PERIOD', 'COMMA'}:
            st.playing, st.instant = False, False
            delta = (1.0 / total) if event.shift else 0.02
            st.frac = max(0.0, min(1.0, st.frac + (delta if t == 'PERIOD' else -delta)))
        elif t in {'EQUAL', 'NUMPAD_PLUS'}:
            st.sweep_secs = max(1.0, st.sweep_secs / 1.4)
        elif t in {'MINUS', 'NUMPAD_MINUS'}:
            st.sweep_secs = min(40.0, st.sweep_secs * 1.4)
        elif t == 'I':
            st.instant = not st.instant
            if not st.instant:
                st.frac = 0.0
        elif t == 'V':
            st.wrong = not st.wrong
        elif t == 'X':
            st.key_mode = "FARTHEST" if st.key_mode == "CENTROID" else "CENTROID"
        elif t == 'B':
            st.cull = not st.cull
        elif t == 'C':
            st.color_mode = (st.color_mode + 1) % 3
        elif t == 'L':
            st.edges = not st.edges
        elif t == 'T':
            st.canvas_mode = (st.canvas_mode + 1) % 3
        elif t == 'H':
            st.help = not st.help
        elif t == 'P':
            self._print_table()
        elif t == 'F':
            st.auto = not st.auto
            self.leg_from = None
        elif t == 'N':
            st.ambient = not st.ambient
        elif t == 'G':
            try:
                self.space.shading.type = 'MATERIAL' if self.space.shading.type == 'SOLID' else 'SOLID'
                if self.space.shading.type == 'MATERIAL':
                    self.space.shading.use_scene_lights = True
                    self.space.shading.use_scene_world = True
            except Exception:
                pass
        elif t == 'M':
            st.mouse_look = not st.mouse_look
            try:
                context.window.cursor_set('NONE' if st.mouse_look else 'DEFAULT')
            except Exception:
                pass
            if st.mouse_look:
                self._warp(context)
        return {'RUNNING_MODAL'}


bpy.utils.register_class(VIEW3D_OT_painter_walkthrough)
NS["painter_op_cls"] = VIEW3D_OT_painter_walkthrough


# =============================================================================
#  11. AUTO-START (Layout workspace -> maximise viewport -> start walkthrough)
# =============================================================================
def _find_view3d(win):
    for area in win.screen.areas:
        if area.type == 'VIEW_3D':
            for r in area.regions:
                if r.type == 'WINDOW':
                    return area, r
    return None, None


def _stage3():
    try:
        win = bpy.context.window_manager.windows[0]
        area, region = _find_view3d(win)
        if area is None:
            raise RuntimeError("no 3D viewport")
        with bpy.context.temp_override(window=win, area=area, region=region):
            bpy.ops.view3d.painter_walkthrough('INVOKE_DEFAULT')
        print("Painter walkthrough started. Press H for help, Esc to exit.")
    except Exception as exc:
        print("Auto-start failed (%s). Hover the 3D viewport, press F3 and search "
              "'Start Painter Walkthrough'." % exc)
    return None


def _stage2():
    try:
        win = bpy.context.window_manager.windows[0]
        area, region = _find_view3d(win)
        if area is not None:
            with bpy.context.temp_override(window=win, area=area, region=region):
                bpy.ops.screen.screen_full_area(use_hide_panels=True)
    except Exception as exc:
        print("Could not maximise the viewport (%s) - continuing." % exc)
    bpy.app.timers.register(_stage3, first_interval=0.5)
    return None


def _stage1():
    try:
        win = bpy.context.window_manager.windows[0]
        ws = bpy.data.workspaces.get("Layout")
        if ws is not None:
            win.workspace = ws
    except Exception:
        pass
    bpy.app.timers.register(_stage2, first_interval=0.5)
    return None


bpy.app.timers.register(_stage1, first_interval=0.4)

print("=" * 70)
print("Indoor navigation scene built. Painter's walkthrough starts automatically.")
print("Controls: WASD walk | mouse look | F autopilot | SPACE pause | V wrong order")
print("          X sort key | B cull | C colours | L edges | T canvas | H help | Esc exit")
print("Animation: Ctrl+F12 renders the 240-frame animation (fans, doors, orb).")
print("=" * 70)
