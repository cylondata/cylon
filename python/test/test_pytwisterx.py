no_twisterx = False
try:
    import pytwisterx
except ImportError:
    no_twisterx = True

if no_twisterx:
    print("No Pytwisterx installation found!")
else:
    print("Pytwisterx Installed!")
