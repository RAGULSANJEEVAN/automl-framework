class TargetSelector:

    def __init__(self, df):
        self.df = df

    def select_target(self, target):

        # No target selected -> Unsupervised Learning
        if target is None:
            return None

        target = str(target).strip()

        if target == "":
            return None

        if target not in self.df.columns:
            raise ValueError(f"'{target}' is not a valid column.")

        return target