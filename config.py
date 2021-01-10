class config: 
    def __init__(self, INTENSITIES, TYPE=0, NUM_HEART_BEATS=2, NUM_SDFS=2):
        '''
        Define the Environment parameters of CT setup
        '''
        self.IMAGE_RESOLUTION = 64              # Resolution of the CT image
        self.GANTRY_VIEWS_PER_ROTATION = 720     # Number of views that the gantry clicks in a single 360 degree rotation
        self.HEART_BEAT_PERIOD = 1000            # Time (ms) it takes the heart to beat once
        self.GANTRY_ROTATION_PERIOD = 275        # Time (ms) it takes for the gantry to complete a single 360 degree rotation
        self.NUM_HEART_BEATS = NUM_HEART_BEATS   # Number of heart beats during the time HEART_BEAT_PERIOD
        self.INTENSITIES = INTENSITIES
        '''
        Parameters for defining experimental setup
        '''
        if TYPE==0:
        # To run gantry for a single 360 degree rotation
            self.TOTAL_CLICKS = self.GANTRY_VIEWS_PER_ROTATION
            self.THETA_MAX = 360
            self.GANTRY2HEART_SCALE = (1/self.THETA_MAX)*(self.GANTRY_ROTATION_PERIOD/self.HEART_BEAT_PERIOD)
        
        elif TYPE==1:
        # Otherwise, to run gantry for a single heart beat
            self.TOTAL_CLICKS = int(self.GANTRY_VIEWS_PER_ROTATION * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD)/self.NUM_HEART_BEATS)
            self.THETA_MAX = int(360 * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD)/self.NUM_HEART_BEATS)
            self.GANTRY2HEART_SCALE = 1/(self.THETA_MAX*self.NUM_HEART_BEATS)
        
        elif TYPE==2:
        # Lastly, if you wish to run gantry to capture N heart beats then
            self.TOTAL_CLICKS = int(self.GANTRY_VIEWS_PER_ROTATION * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD))
            self.THETA_MAX = int(360 * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD))
            self.GANTRY2HEART_SCALE = 1/(self.THETA_MAX)
        
        '''
        NeuralCT Hyper parameters
        '''
        self.SDF_SCALING = 0.5*self.IMAGE_RESOLUTION  # Factor to scale NeuralCT's output to match G.T. SDF range of values
        self.BATCH_SIZE=20                       # Number of projections used in a single training iterations
        self.NUM_SDFS = NUM_SDFS