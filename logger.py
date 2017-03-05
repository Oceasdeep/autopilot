class ResultLogger(object):
    """Class for handling result logging to CSV files"""

    def __init__(self, path='logs/runlog.csv'):
        """Opens a new CSV log file for writing and adds a header row

        Args:
            path (str, optional): Path to the log file.
        """

        # Open log file for writing
        self.log = open(path, 'w')

        # Write initial log entry
        self.log.write('Index,Time,Time_Diff,Output\n')


    def write(self, i, t, delta_t, output):
        """Writes a single entry to the log file

        Args:
            i (int): Image index.
            t (float): Timestamp in seconds
            delta_t (float): Time difference in seconds between this and
                previous timestamps
            output (float): Output of the Inference

        """

        self.log.write('%d,%f,%f,%f\n' % (i, t, delta_t, output))

    def __del__(self):
        self.log.close()
