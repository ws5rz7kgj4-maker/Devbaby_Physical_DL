"""Headless, procedural 3D rigid-body world, separated observations/annotations."""
import math
from collections import deque
import numpy as np
import pybullet as p
from .audio import features, impact
from .utils import derived_seed

COLORS = {"red": (.85,.08,.06,1), "blue": (.06,.2,.85,1),
          "green": (.06,.65,.15,1), "yellow": (.93,.8,.05,1)}
SHAPES = ("ball", "block", "cylinder")
COMBOS = [(c,s) for c in COLORS for s in SHAPES]
HELDOUT = {("red","ball"), ("blue","block"), ("green","cylinder")}
TRAIN_COMBOS = [x for x in COMBOS if x not in HELDOUT]
CONDITIONS = ("iid", "combination", "layout", "voice", "physics", "joint")
ACTION_DIM = 5
PROPRIO_DIM = 10

class World:
    def __init__(self, cfg, speech, seed, condition="train", objects=None, caregiver=True):
        self.cfg, self.speech, self.seed, self.condition = cfg, speech, seed, condition
        self.cid = p.connect(p.DIRECT)
        self.rng = np.random.default_rng(derived_seed(seed, "physics"))
        self.language_rng = np.random.default_rng(derived_seed(seed, "language"))
        self.sound_rng = np.random.default_rng(derived_seed(seed, "sound"))
        self.dt = 1 / cfg.control_hz
        self.n_audio = cfg.sample_rate // cfg.control_hz
        self.step_index = 0
        self.caregiver = caregiver
        self.closed = False
        self.constraint = None
        self.held_id = None
        self.yaw = float(self.rng.uniform(-.12,.12))
        self.pending_audio = np.zeros(cfg.sample_rate * 8, np.float32)
        self.pending_speech = []
        self.utterances = []
        self.events = []
        self.previous_contacts = set()
        self.object_ids = []
        self.objects = []
        self.static_geometry = []
        self.history = []
        self.next_speech = int(self.language_rng.integers(0, 7))
        self.next_disturbance = int(self.rng.integers(4, 20))
        self.focal_index = 0
        self._build(objects)

    def call(self, method, *args, **kwargs):
        return getattr(p, method)(*args, physicsClientId=self.cid, **kwargs)

    def _box(self, half, pos, rgba, mass=0):
        cs = self.call("createCollisionShape", p.GEOM_BOX, halfExtents=half)
        vs = self.call("createVisualShape", p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        bid=self.call("createMultiBody", baseMass=mass, baseCollisionShapeIndex=cs,
                      baseVisualShapeIndex=vs, basePosition=pos)
        self.static_geometry.append(dict(id=bid,shape="box",half_extents=half,position=pos,rgba=rgba,mass=mass))
        return bid

    def _build(self, requested):
        self.call("resetSimulation")
        novel_dynamics=self.condition in {"physics","joint","new_dynamics","joint_physics"}
        self.gravity = -9.81 if not novel_dynamics else -6.0
        self.hand_mass=float(self.rng.uniform(.08,.12)) if self.cfg.motor_domain_randomization else .1
        self.motor_gain=float(self.rng.uniform(30,40)) if self.cfg.motor_domain_randomization else 35.
        self.motor_damping=float(self.rng.uniform(2.5,3.5)) if self.cfg.motor_domain_randomization else 3.
        self.motor_step=float(self.rng.uniform(.05,.06)) if self.cfg.motor_domain_randomization else .055
        self.motor_delay=0
        if novel_dynamics and self.cfg.motor_domain_randomization:
            self.hand_mass=float(self.rng.uniform(.20,.28))
            self.motor_gain=float(self.rng.uniform(14,20))
            self.motor_damping=float(self.rng.uniform(3.8,4.8))
            self.motor_step=float(self.rng.uniform(.035,.045))
        if self.condition in {"new_delay","joint_physics"}:
            self.motor_delay=int(self.rng.integers(2,4))
        self.action_queue=deque([np.zeros(ACTION_DIM,np.float32) for _ in range(self.motor_delay)])
        self.call("setGravity", 0, 0, self.gravity)
        self.call("setTimeStep", 1/self.cfg.physics_hz)
        self.call("setPhysicsEngineParameter", deterministicOverlappingPairs=1, numSolverIterations=60)
        novel = self.condition in {"layout","joint","new_layout","joint_physics"}
        theme = int(self.rng.integers(0,3)) if not novel else 3
        floors = [(.62,.49,.35,1),(.48,.58,.62,1),(.6,.6,.55,1),(.25,.3,.42,1)]
        self._box([1.0,.8,.035], [.35,0,-.035], floors[theme])
        self._box([.035,.8,.45], [1.35,0,.4], [.7,.72,.75,1])
        self._box([1,.025,.25], [.35,-.8,.22], [.7,.7,.7,1])
        self._box([1,.025,.25], [.35,.8,.22], [.7,.7,.7,1])
        # Different layouts; there is no semantic ordering of object slots.
        self.theme = theme
        self.props = []
        if theme in {1,3}:
            self.props.append(self._box([.12,.09,.035], [.9,-.45,.035], [.55,.45,.6,1]))
        if theme in {2,3}:
            self.props.append(self._box([.06,.15,.12], [.9,.48,.12], [.4,.5,.4,1]))
        if self.condition in {"new_layout","joint_physics"}:
            self.props.append(self._box([.04,.16,.09], [.48,-.08,.09], [.45,.43,.38,1]))
        n = int(self.rng.integers(self.cfg.objects_min, self.cfg.objects_max+1)) if requested is None else len(requested)
        options = COMBOS if self.condition in {"combination","joint"} else TRAIN_COMBOS
        chosen = [options[int(self.rng.integers(len(options)))] for _ in range(n)] if requested is None else requested
        positions = []
        for i, (color, shape) in enumerate(chosen):
            for _ in range(100):
                pos = [float(self.rng.uniform(.15,.8)), float(self.rng.uniform(-.4,.4)), .13]
                if self.condition in {"new_layout","joint_physics"} and abs(pos[0]-.48)<.14 and abs(pos[1]+.08)<.25:
                    continue
                if all(np.linalg.norm(np.array(pos[:2])-np.array(q[:2])) > .16 for q in positions):
                    break
            positions.append(pos)
            size = float(self.rng.uniform(.045,.075))
            material = str(self.rng.choice(["wood","metal","rubber"]))
            mass = float(self.rng.uniform(.05,.25))
            friction = float(self.rng.uniform(.25,.8))
            if novel_dynamics:
                mass *= 3
                friction = float(self.rng.uniform(.02,.1))
            kw = {"radius":size} if shape == "ball" else ({"halfExtents":[size]*3} if shape=="block" else {"radius":size,"height":size*1.7})
            geom = {"ball":p.GEOM_SPHERE,"block":p.GEOM_BOX,"cylinder":p.GEOM_CYLINDER}[shape]
            cs = self.call("createCollisionShape", geom, **kw)
            vkw = dict(kw)
            if "height" in vkw:
                vkw["length"] = vkw.pop("height")
            vs = self.call("createVisualShape", geom, rgbaColor=COLORS[color], **vkw)
            bid = self.call("createMultiBody", baseMass=mass, baseCollisionShapeIndex=cs,
                            baseVisualShapeIndex=vs, basePosition=pos)
            self.call("changeDynamics", bid, -1, lateralFriction=friction, restitution=.3,
                      rollingFriction=.008, linearDamping=.05, angularDamping=.08)
            self.object_ids.append(bid)
            self.objects.append(dict(id=bid,color=color,shape=shape,size=size,material=material,
                                     mass=mass,friction=friction,restitution=.3,rolling_friction=.008,
                                     linear_damping=.05,angular_damping=.08,rgba=COLORS[color],initial_position=pos))
        # A simple infant effector, not an anatomically realistic infant body.
        cs = self.call("createCollisionShape", p.GEOM_SPHERE, radius=.038)
        vs = self.call("createVisualShape", p.GEOM_SPHERE, radius=.038, rgbaColor=[.95,.69,.47,1])
        pos = [float(self.rng.uniform(.05,.7)), float(self.rng.uniform(-.3,.3)), .22]
        if self.condition in {"new_layout","joint_physics"}:
            pos[2]=.34
        self.hand = self.call("createMultiBody", baseMass=self.hand_mass,baseCollisionShapeIndex=cs,baseVisualShapeIndex=vs,basePosition=pos)
        self.target = np.asarray(pos, dtype=float)
        self.call("changeDynamics",self.hand,-1,linearDamping=.9,angularDamping=.99)
        # Visible caregiver head and pointing marker (perceptible social cue).
        head = self.call("createVisualShape",p.GEOM_SPHERE,radius=.095,rgbaColor=[.8,.6,.5,1])
        self.adult = self.call("createMultiBody",baseVisualShapeIndex=head,basePosition=[1.05,.5,.6])
        marker = self.call("createVisualShape",p.GEOM_SPHERE,radius=.018,rgbaColor=[1,.85,.65,1])
        self.pointer = self.call("createMultiBody",baseVisualShapeIndex=marker,basePosition=[1.05,.5,.4])
        for _ in range(30):
            self._physics_tick()
        self.previous_contacts = set()
        self.events = []
        self.pending_audio[:] = 0

    def _physics_tick(self):
        pos, _ = self.call("getBasePositionAndOrientation", self.hand)
        vel, _ = self.call("getBaseVelocity",self.hand)
        force = np.clip(self.motor_gain*(self.target-np.array(pos))-self.motor_damping*np.array(vel),-8,8)
        force[2] -= self.hand_mass*self.gravity
        self.call("applyExternalForce",self.hand,-1,force.tolist(),pos,p.WORLD_FRAME)
        self.call("stepSimulation")

    def _add_audio(self, wav, offset=0):
        n = min(len(wav), len(self.pending_audio)-offset)
        if n > 0:
            self.pending_audio[offset:offset+n] += wav[:n]

    def _speak(self):
        r = self.language_rng
        self.focal_index = int(r.integers(len(self.objects)))
        obj = self.objects[self.focal_index]
        grounded = bool(r.random() < self.cfg.correspondence)
        # Always consume identical RNG draws in both correspondence conditions.
        # Random naming changes the sound, not the pointing/timing/physical experience.
        options = TRAIN_COMBOS if self.condition not in {"combination","joint"} else COMBOS
        alternative = options[int(r.integers(len(options)))]
        alternative_event = str(r.choice(["moving","falling","still"]))
        if grounded:
            color, shape = obj["color"], obj["shape"]
        else:
            # Independent draw, not deliberately wrong: at q=0 there is no reliable mapping.
            color, shape = alternative
        templates = ["{c} {s}", "look at the {c} {s}", "there is a {c} {s}"]
        text = str(r.choice(templates)).format(c=color,s=shape)
        spoken_event=None
        if r.random()<.4:
            vel,_=self.call("getBaseVelocity",obj["id"])
            actual="falling" if vel[2]<-.1 else ("moving" if np.linalg.norm(vel)>.04 else "still")
            spoken_event=actual if grounded else alternative_event
            text=f"the {color} {shape} is {spoken_event}"
        voice = "en-gb-scotland" if self.condition in {"voice","joint"} else str(r.choice(["en-us","en"]))
        speed = int(r.choice([175,195,215]))
        pitch = int(r.choice([35,50,65]))
        wav = self.speech.synthesize(text,voice,speed,pitch)
        offset = self.cfg.caregiver_delay_steps*self.n_audio
        self._add_audio(wav,offset)
        event = dict(type="utterance",start_s=self.step_index*self.dt+offset/self.cfg.sample_rate,
                     end_s=self.step_index*self.dt+offset/self.cfg.sample_rate+len(wav)/self.cfg.sample_rate,
                     text=text,voice=voice,speed=speed,pitch=pitch,related=grounded,
                     intended_referent=obj["id"] if grounded else None,
                     spoken_color=color,spoken_shape=shape,pointed_object=obj["id"],
                     spoken_event=spoken_event,
                     pairing_available_to_learner=False)
        self.utterances.append(event)
        # The same exogenous schedule for q=0 and q=1; word duration cannot
        # change the later scene or the caregiver's visible pointing behavior.
        self.next_speech = self.step_index + int(r.integers(28,37))

    def observe(self, audio=None):
        im, mask = self.render()
        self.latest_mask = mask
        pos,_ = self.call("getBasePositionAndOrientation",self.hand)
        vel,_ = self.call("getBaseVelocity",self.hand)
        contact = any(c[2] in self.object_ids for c in self.call("getContactPoints",bodyA=self.hand))
        proprio = np.asarray([*pos,*np.clip(vel,-2,2),float(self.closed),float(contact),self.yaw,float(self.constraint is not None)],np.float32)
        if audio is None:
            audio = np.zeros(self.n_audio,np.float32)
        return {"rgb":im,"audio":features(audio,self.cfg.sample_rate),"proprio":proprio}

    def render(self, size=None):
        size = size or self.cfg.image_size
        eye = [-.62,0,.57]
        target = [.5,math.tan(self.yaw)*1.12,.05]
        view = p.computeViewMatrix(eye,target,[0,0,1])
        proj = p.computeProjectionMatrixFOV(62,1,.02,5)
        self.latest_camera=dict(eye=eye,target=target,up=[0,0,1],fov_vertical_deg=62,near=.02,far=5,
                                view_matrix_column_major=list(view),projection_matrix_column_major=list(proj))
        out = self.call("getCameraImage",size,size,viewMatrix=view,projectionMatrix=proj,
                        renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                        shadow=0,lightDirection=[-1,-1,3])
        return np.asarray(out[2],np.uint8).reshape(size,size,4)[:,:,:3].copy(), np.asarray(out[4]).reshape(size,size)

    def step(self, action):
        action = np.asarray(action,np.float32)
        assert action.shape == (ACTION_DIM,) and np.isfinite(action).all()
        action = np.clip(action,-1,1)
        commanded_action=action.copy()
        self.action_queue.append(action)
        action=self.action_queue.popleft()
        self.events = []
        if self.caregiver and self.step_index >= self.next_speech:
            self._speak()
        if self.caregiver:
            pos,_ = self.call("getBasePositionAndOrientation",self.object_ids[self.focal_index])
            self.call("resetBasePositionAndOrientation",self.pointer,[pos[0],pos[1],pos[2]+.13],[0,0,0,1])
        self.target = np.clip(self.target + action[:3]*self.motor_step,[-.1,-.62,.045],[1.1,.62,.65])
        self.yaw = float(np.clip(self.yaw+action[4]*.045,-.5,.5))
        if action[3] < -.4:
            self.closed = False
            if self.constraint is not None:
                self.call("removeConstraint",self.constraint)
                self.events.append(dict(type="release",actor="infant",object=self.held_id))
                self.constraint,self.held_id = None,None
        elif action[3] > .4:
            self.closed = True
        if self.closed and self.constraint is None:
            hpos,hori = self.call("getBasePositionAndOrientation",self.hand)
            for obj in self.objects:
                opos,oori = self.call("getBasePositionAndOrientation",obj["id"])
                if np.linalg.norm(np.asarray(hpos)-np.asarray(opos)) < obj["size"]+.048:
                    inv=p.invertTransform(hpos,hori)
                    rel=p.multiplyTransforms(*inv,opos,oori)
                    self.constraint=self.call("createConstraint",self.hand,-1,obj["id"],-1,p.JOINT_FIXED,[0,0,0],rel[0],[0,0,0],parentFrameOrientation=rel[1])
                    self.held_id=obj["id"]
                    self.events.append(dict(type="grasp",actor="infant",object=obj["id"]))
                    break
        if self.caregiver and self.step_index >= self.next_disturbance:
            obj = self.objects[int(self.rng.integers(len(self.objects)))]
            pos,_=self.call("getBasePositionAndOrientation",obj["id"])
            # One physics-step force equivalent to a bounded random velocity impulse.
            f=obj["mass"]*self.cfg.physics_hz*self.rng.uniform([-.8,-.8,0],[.8,.8,2])
            self.call("applyExternalForce",obj["id"],-1,f.tolist(),pos,p.WORLD_FRAME)
            self.events.append(dict(type="external_push",actor="caregiver",object=obj["id"],force=f.tolist()))
            self.next_disturbance=self.step_index+int(self.rng.integers(10,30))
        # New impacts, including their within-interval timing, drive synthetic sound.
        for sub in range(self.cfg.physics_hz//self.cfg.control_hz):
            self._physics_tick()
            now=set()
            for c in self.call("getContactPoints"):
                pair=tuple(sorted([c[1],c[2]]))
                now.add(pair)
                if pair not in self.previous_contacts and c[9]>.05:
                    oid=next((bid for bid in pair if bid in self.object_ids),None)
                    if oid is not None:
                        obj=self.objects[self.object_ids.index(oid)]
                        offset=round(sub/self.cfg.physics_hz*self.cfg.sample_rate)
                        self._add_audio(impact(obj["material"],c[9],self.cfg.sample_rate,self.sound_rng),offset)
                        self.events.append(dict(type="impact",bodies=list(pair),normal_force=float(c[9]),offset_s=sub/self.cfg.physics_hz))
            self.previous_contacts=now
        audio=np.clip(self.pending_audio[:self.n_audio],-1,1).copy()
        self.pending_audio[:-self.n_audio]=self.pending_audio[self.n_audio:]
        self.pending_audio[-self.n_audio:]=0
        self.step_index+=1
        obs=self.observe(audio)
        annotation=self.annotation(commanded_action)
        annotation["applied_action"]=action.tolist()
        return obs,audio,annotation

    def annotation(self, action=None):
        objs=[]
        mask=getattr(self,"latest_mask",np.zeros((1,1),int))
        for obj in self.objects:
            pos,rot=self.call("getBasePositionAndOrientation",obj["id"])
            vel,ang=self.call("getBaseVelocity",obj["id"])
            ys,xs=np.where((mask & ((1<<24)-1))==obj["id"])
            objs.append({**obj,"position":list(pos),"orientation_xyzw":list(rot),"velocity":list(vel),
                         "angular_velocity":list(ang),"visible_pixels":int(len(xs)),
                         "bbox_xyxy":None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
                         "moving":bool(np.linalg.norm(vel)>.04),"falling":bool(vel[2]<-.1)})
        return dict(time_s=self.step_index*self.dt,objects=objs,events=list(self.events),
                    action=None if action is None else np.asarray(action).tolist(),
                    hand_position=list(self.call("getBasePositionAndOrientation",self.hand)[0]),
                    motor_target_position=self.target.tolist(),camera=getattr(self,"latest_camera",None),
                    caregiver_pointer_position=list(self.call("getBasePositionAndOrientation",self.pointer)[0]),
                    head_yaw=self.yaw,grasped_object=self.held_id)

    def metadata(self):
        return dict(seed=self.seed,condition=self.condition,theme=self.theme,gravity=self.gravity,
                    objects=self.objects,static_geometry=self.static_geometry,
                    effector=dict(id=self.hand,radius=.038,mass=self.hand_mass,linear_damping=.9,angular_damping=.99,
                                  servo_gain=self.motor_gain,servo_damping=self.motor_damping,
                                  command_step_m=self.motor_step,command_delay_steps=self.motor_delay),
                    caregiver_avatar=dict(head_id=self.adult,head_radius=.095,head_position=[1.05,.5,.6],pointer_id=self.pointer,pointer_radius=.018),
                    control_hz=self.cfg.control_hz,physics_hz=self.cfg.physics_hz,
                    sample_rate=self.cfg.sample_rate,coordinates="world x/y horizontal, z up; metres, seconds, kg",
                    embodiment="3D point effector with proximity grasp reflex; no anatomical infant simulation")

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
