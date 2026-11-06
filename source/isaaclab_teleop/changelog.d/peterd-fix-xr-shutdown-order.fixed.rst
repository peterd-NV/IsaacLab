* Fixed Kit XR teleoperation shutdown to release its OpenXR instance before stopping
  the CloudXR runtime, avoiding broken-pipe and instance-lost errors after recording.
