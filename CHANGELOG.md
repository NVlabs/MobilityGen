# Changelog

# main

- Added example for parquet conversion (to support X-Mobility training)
- Added robot linear and angular velocity to state (to support X-Mobility training)
- Fixed bug when replay rendering does not include segmentation info
- Added support for surface normals image in replay rendering
- Added support for instance ID segmentation rendering
- Added camera world pose to state

# ref-4.5

Release targeting Isaac Sim 4.5

- Updated to use isaacsim.* rather than omni.* import calls
- Added minor fixes to API discrepancies
- Added required extension loading for replay rendering
- Updated README to include setup instructions for Isaac Sim 4.5

# ref-4.2

Release targeting Isaac Sim 4.2

- Added modular state capture
- Added state writer class
- Added state reader class
- Added support for H1
- Added support for Jetbot
- Added support for Spot 
- Added support for Carter
- Added keyboard teleop scenario
- Added gamepad teleop scenario
- Added path following scenario
- Added random acceleration scenario
- Added initial OV extension for launching scenarios and recording data
- Added replay rendering scripts
- Added data loading examples
- Added workflow documentation
