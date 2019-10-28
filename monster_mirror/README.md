Performance notes...

* Total processing with 2 faces, on ~2MP image: ~600ms
* Haar face finder: 230ms  -- opportunity
* per-face...
- * image prep: 40ms
- * CNN time (per face): 19ms
- * up-resize: 23ms
- * blend: 100ms  -- opportunity!


