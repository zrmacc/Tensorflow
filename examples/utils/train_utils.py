import os

# ----------------------------------------------------------------------------

def clear_checkpoint_dir(check_dir, prefix):
    """Clears contents of a checkpoint directory containing a prefix."""
    dir_contents = os.listdir(check_dir)
    matched = [file for file in dir_contents if prefix in file]
    for file in matched:
        target = os.path.join(check_dir, file)
        os.remove(target)
    
    return None