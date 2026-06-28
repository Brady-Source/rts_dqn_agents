import numpy as np
from dataclasses import dataclass

GRID_SIZE = 15

@dataclass
class AgentState:
    x: int
    y: int
    alive: bool = True
    stat: int = 1 # This is for Fighters
    
class FarmersFighterEnv:
    def __init__(self, max_steps=200):
        self.max_steps = max_steps
        self.step_count = 0
        
        # Defining the agents
        self.team1_farmers=None
        self.team1_fighter=None
        self.team2_farmer=None
        self.team2_fighter=None

        # Crops and buffs
        self.crops=[]
        self.buffs=[]
        
    def reset(self):
        self.step_count = 0
        self.crops = []
        self.buffs = []
        
        # Assigning the spawn zones for the agents
        self.team1_farmer = self._spawn_in_box(0,5,0,5)
        self.team1_fighter = self._spawn_in_box(0,5,0,5)
        self.team2_farmer = self._spawn_in_box(9,14,9,14)
        self.team2_fighter = self._spawn_in_box(9,14,9,14)
        
        #Agents stats
        self.team1_fighter.stat = 1
        self.team2_fighter.stat = 1
        
        obs = self._get_observations()
        return obs
    
    #Generating random spawn location
    def _spawn_in_box(self, row_min, row_max, col_min, col_max):
        x = np.random.randint(row_min, row_max + 1)
        y = np.random.randint(col_min, col_max + 1)
        return AgentState(x=x, y=y, alive=True, stat=1)
    
    def _maybe_spawn_crops(self):
        # Every 5 steps spawn up to 2 crops, 1 in each North/South half respecting a max of 3 per half.
        if self.step_count % 5 != 0:
            return
        
        # Top half
        top_crops = [c for c in self.crops if c[0] <= 6]
        if len(top_crops) < 3:
            x_top = np.random.randint(0, 7)
            y_top = np.random.randint(0, GRID_SIZE)
            self.crops.append((x_top, y_top))
            
        # Bottom half
        bottom_crops = [c for c in self.crops if c[0] >= 8]
        if len(bottom_crops) < 3:
            x_bot = np.random.randint(8, GRID_SIZE)
            y_bot = np.random.randint(0, GRID_SIZE)
            self.crops.append((x_bot, y_bot))
            
    def _maybe_spawn_buff(self):
        # Every 10 steps spawn a central buff that will last 5 steps when picked up.
        if self.step_count % 10 != 0:
            return
        # Defining the central location to spawn the buff.
        x = np.random.randint(4, 11)
        y = np.random.randint(4, 11)
        self.buffs.append({'x': x, 'y': y, 'expires': self.step_count + 5})
        
    def _update_buffs(self):
        self.buffs = [b for b in self.buffs if b['expires'] > self.step_count]
        
    def _check_done(self):
        if self.team1_farmer.alive == False and self.team1_fighter.alive == False:
            print("Team 2")
            return True
        elif self.team2_farmer.alive == False and self.team2_fighter.alive == False:
            print("Team 1")
            return True
        elif self.step_count >= self.max_steps:
            print("No Winners")
            return True
        else:
            return False
        
    # Defining the Step process
    def step(self, actions):
        self.step_count += 1
        
        # Movement
        self._move_agent(self.team1_farmer, actions['t1_farmer'])
        self._move_agent(self.team1_fighter, actions['t1_fighter'])
        self._move_agent(self.team2_farmer, actions['t2_farmer'])
        self._move_agent(self.team2_fighter, actions['t2_fighter'])
        
        # Spawn crops and buffs
        self._maybe_spawn_crops()
        self._maybe_spawn_buff()
        self._update_buffs()
        
        # Rewards
        rewards = self._compute_rewards(actions)
        
        # Observations
        obs = self._get_observations()
        
        # Is the episode complete
        done = self._check_done()
        
        info = {}
        return obs, rewards, done, info
    
    def _move_agent(self, agent, action):
        if not agent.alive:
            return
        dx, dy = 0, 0
        if action == 1:
            dx = -1
        elif action ==2:
            dx = 1
        elif action ==3:
            dy = -1
        elif action ==4:
            dy = 1
        new_x = np.clip(agent.x +dx, 0, GRID_SIZE -1)
        new_y = np.clip(agent.y + dy, 0, GRID_SIZE -1)
        agent.x, agent.y = new_x, new_y
        
    def _harvest_crops(self, agent, rewards, agent_key, fighter_agent):
        if not agent.alive:
            return rewards
        
        harvested = []
        #Checking if the agent is at a crop
        for (x, y) in self.crops:
            if agent.x == x and agent.y == y:
                harvested.append((x, y))
                
        # Remove the crop once harvested
        if harvested:
            for h in harvested:
                self.crops.remove(h)
                
            rewards[agent_key] += 5.0
            if fighter_agent is not None and fighter_agent.alive:
                fighter_agent.stat += len(harvested)
        return rewards
    
    def _pick_up_buff(self, fighter_agent, rewards, agent_key):
        if not fighter_agent.alive:
            return rewards
        
        picked = []
        for b in self.buffs:
            if fighter_agent.x == b['x'] and fighter_agent.y == b['y']:
                picked.append(b)
             
        # Rewards for picking up buff   
        if picked:
            for b in picked:
                self.buffs.remove(b)
            rewards[agent_key] += 5.0
            fighter_agent.stat += 5
        return rewards
    
    def _resolve_combat(self, rewards):
        t1_f = self.team1_fighter
        t1_farmer = self.team1_farmer
        t2_f = self.team2_fighter
        t2_farmer = self.team2_farmer

        # The fighters combat
        
        if t1_f.alive and t2_f.alive:
            if t1_f.x == t2_f.x and t1_f.y == t2_f.y:
                if t1_f.stat > t2_f.stat:
                    t2_f.alive = False
                    rewards['t1_fighter'] += 30.0
                    rewards['t2_fighter'] -= 10.0
                elif t2_f.stat > t1_f.stat:
                    t1_f.alive = False
                    rewards['t2_fighter'] += 30.0
                    rewards['t1_fighter'] -= 10.0
                else:
                    t1_f.alive = True
                    t2_f.alive = True

        # Reward for Team 1 fighter eliminating Team 2 Farmer
        if t1_f.alive and t2_farmer.alive:
            if t1_f.x == t2_farmer.x and t1_f.y == t2_farmer.y:
                t2_farmer.alive = False
                rewards['t1_fighter'] += 20.0
                rewards['t2_farmer'] -= 5.0
                
        # Reward for Team 2 fighter eliminating Team 1 Farmer
        if t2_f.alive and t1_farmer.alive:
            if t2_f.x == t1_farmer.x and t2_f.y == t1_farmer.y:
                t1_farmer.alive = False
                rewards['t2_fighter'] += 20.0
                rewards['t1_farmer'] -= 5.0
                
        t1_all_dead = (not t1_f.alive) and (not t1_farmer.alive)
        t2_all_dead = (not t2_f.alive) and (not t2_farmer.alive)
        
        # Rewards for eliminating the other team
        if t1_all_dead and not t2_all_dead:
            rewards['t2_fighter'] += 100
        elif t2_all_dead and not t1_all_dead:
            rewards['t1_fighter'] += 100
        
        return rewards
        
    def _compute_rewards(self, actions):
        rewards = {
            't1_farmer': -0.3,
            't1_fighter': -0.3,
            't2_farmer': -0.3,
            't2_fighter': -0.3,
        } # the time penalty
        
        # Rewards for harvesting when in range
        rewards = self._harvest_crops(self.team1_farmer, rewards, 't1_farmer', self.team1_fighter)
        rewards = self._harvest_crops(self.team2_farmer, rewards, 't2_farmer', self.team2_fighter)
        
        # Not sure if fighter should have this ability but they can also harvest crops at a slower speed (2 steps instead of the famers 1 step)
        # rewards = self._harvest_crops(self.team1_fighter, rewards, 't1_fighter', self.team1_fighter, fighter=True)
        # rewards = self._harvest_crops(self.team2_fighter, rewards, 't2_fighter', self.team2_fighter, fighter=True)
        
        # Buff pickups for fighters
        rewards = self._pick_up_buff(self.team1_fighter, rewards, 't1_fighter')
        rewards = self._pick_up_buff(self.team2_fighter, rewards, 't2_fighter')
        
        # Combat rewards
        rewards = self._resolve_combat(rewards)
        
        return rewards
        
    import torch

    def _get_observations(self):
        obs = {}
        obs['t1_farmer']  = self._build_obs_for_agent(self.team1_farmer,  view_range=3)
        obs['t1_fighter'] = self._build_obs_for_agent(self.team1_fighter, view_range=2)
        obs['t2_farmer']  = self._build_obs_for_agent(self.team2_farmer,  view_range=3)
        obs['t2_fighter'] = self._build_obs_for_agent(self.team2_fighter, view_range=2)
        return obs
    
    def _build_obs_for_agent(self, agent, view_range):
        window = 2 * view_range + 1
        grid = np.zeros((6, window, window), dtype = np.float32)
        
        if not agent.alive:
            return grid
        
        ax, ay = agent.x, agent.y
        
        def to_local(wx, wy):
            lx = wx - ax + view_range
            ly = wy - ay + view_range
            return lx, ly
        
        def in_window(lx, ly):
            return 0 <= lx < window and 0 <= ly < window
        
        grid[0, view_range, view_range] = 1.0
        
        # Channel 1 — t1 fighter
        lx, ly = to_local(self.team1_fighter.x, self.team1_fighter.y)
        if in_window(lx, ly) and self.team1_fighter.alive:
            grid[1, lx, ly] = 1.0

        # Channel 2 — t2 fighter
        lx, ly = to_local(self.team2_fighter.x, self.team2_fighter.y)
        if in_window(lx, ly) and self.team2_fighter.alive:
            grid[2, lx, ly] = 1.0

        # Channel 3 — t1 farmer
        lx, ly = to_local(self.team1_farmer.x, self.team1_farmer.y)
        if in_window(lx, ly) and self.team1_farmer.alive:
            grid[3, lx, ly] = 1.0

        # Channel 4 — t2 farmer
        lx, ly = to_local(self.team2_farmer.x, self.team2_farmer.y)
        if in_window(lx, ly) and self.team2_farmer.alive:
            grid[4, lx, ly] = 1.0

        # Channel 5 — crops
        for (cx, cy) in self.crops:
            lx, ly = to_local(cx, cy)
            if in_window(lx, ly):
                grid[5, lx, ly] = 1.0

        return grid
    