from pin_toggle import toggle, toggle_forever # your file containing the function


def call_arduino():
    """
        sets pin high to let arudion know to actuate the arm.
    """
    toggle(27, 2) # toggles pin on arduino


call_arduino()