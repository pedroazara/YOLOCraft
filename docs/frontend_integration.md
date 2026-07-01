# Frontend Integration — Mob Detection & Segmentation

Spec for the frontend. The user uploads an image; the backend detects Minecraft
mobs and returns, for each one, its class, bounding box, and a segmentation
polygon. The frontend overlays these and highlights the segmented mob.

Flow:

```
upload image  ->  POST /predict  ->  JSON (class + box + polygon)  ->  render / highlight
```

The frontend never runs the model. It only sends the image and draws the JSON.

## API

### Endpoint

```
POST {API_BASE_URL}/predict
Content-Type: multipart/form-data
```

| Field | Type | Description                     |
| ----- | ---- | ------------------------------- |
| file  | File | The image to analyze (jpg/png). |

`API_BASE_URL` must be configurable (env var), e.g. `http://localhost:8000`.
CORS is handled on the backend.

### Response — `200 application/json`

```json
{
  "width": 1152,
  "height": 480,
  "detections": [
    {
      "class": "spider",
      "confidence": 0.41,
      "box": [712.0, 205.4, 1098.7, 430.2],
      "polygon": [[834, 230], [832, 236], [900, 410], [1050, 360]]
    }
  ]
}
```

### Field reference

| Field                    | Type              | Unit / meaning                                                                 |
| ------------------------ | ----------------- | ------------------------------------------------------------------------------ |
| `width`                  | int               | Original image width, in pixels.                                               |
| `height`                 | int               | Original image height, in pixels.                                              |
| `detections`             | array             | One entry per detected mob. Can be empty (`[]`).                               |
| `detections[].class`     | string            | Mob class name (e.g. `"spider"`, `"creeper"`).                                 |
| `detections[].confidence`| float (0.0–1.0)   | Detection confidence.                                                          |
| `detections[].box`       | [x1, y1, x2, y2]  | Bounding box in **original-image pixels**, top-left origin.                    |
| `detections[].polygon`   | array of [x, y]   | Segmentation contour in **original-image pixels**. May be `[]` (no mask).      |

Key rule: **all coordinates are in original-image pixels.** Use `width`/`height`
to map them onto whatever size the image is displayed at (see below).

### States the frontend must handle

- `detections: []` → request succeeded but no mob was found. Show the image and a
  "no mob detected" message. Not an error.
- `polygon: []` on a detection → draw the box only, skip the mask.
- Non-2xx status or network failure → show an error message; do not crash.
- While the request is in flight → show a loading state and disable the upload
  button. Inference can take a few hundred ms to a couple of seconds.

## Consuming the API

```js
async function predict(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}
```

## Rendering — recommended: SVG with `viewBox`

Set the SVG `viewBox` to the original image size. The browser then scales
everything automatically, so you draw with the raw pixel coordinates from the
API — **no manual scaling math, no scaling bugs.** Let the SVG be responsive
with `width: 100%`.

```jsx
<svg viewBox={`0 0 ${data.width} ${data.height}`} style={{ width: "100%" }}>
  <image href={imageUrl} width={data.width} height={data.height} />

  {data.detections.map((d, i) => (
    <g key={i}>
      {d.polygon.length > 0 && (
        <polygon
          points={d.polygon.map((p) => p.join(",")).join(" ")}
          fill="rgba(0,255,0,0.35)"
          stroke="lime"
          strokeWidth="2"
        />
      )}
      <rect
        x={d.box[0]} y={d.box[1]}
        width={d.box[2] - d.box[0]} height={d.box[3] - d.box[1]}
        fill="none" stroke="white" strokeWidth="1"
      />
      <text x={d.box[0]} y={d.box[1] - 4} fill="white" fontSize="14">
        {d.class} {(d.confidence * 100).toFixed(0)}%
      </text>
    </g>
  ))}
</svg>
```

## Rendering — canvas alternative

Canvas needs an explicit scale factor (displayed size ÷ original size):

```js
const scale = canvas.width / data.width;

for (const d of data.detections) {
  if (d.polygon.length) {
    ctx.beginPath();
    d.polygon.forEach(([x, y], i) =>
      i ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale)
    );
    ctx.closePath();
    ctx.fillStyle = "rgba(0,255,0,0.35)";
    ctx.fill();
    ctx.strokeStyle = "lime";
    ctx.stroke();
  }
}
```

## Highlight / spotlight effect

This is the product goal: dim the background and keep the segmented mob at full
brightness, using its polygon as a clip region (canvas):

```js
ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
ctx.fillStyle = "rgba(0,0,0,0.6)";
ctx.fillRect(0, 0, canvas.width, canvas.height);

for (const d of data.detections) {
  if (!d.polygon.length) continue;
  ctx.save();
  ctx.beginPath();
  d.polygon.forEach(([x, y], i) =>
    i ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale)
  );
  ctx.closePath();
  ctx.clip();
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height); // redraw mob at full brightness
  ctx.restore();
}
```

In SVG, the same effect is done with a dark `<rect>` over the whole image plus a
`<mask>`/`<clipPath>` built from the polygons.

## Coordinate system (do not skip)

The most common integration bug is coordinate mismatch. The API always returns
pixels relative to the **original** image. The displayed image is usually a
different size (responsive layout). Two safe options:

1. **SVG `viewBox="0 0 width height"`** — the browser scales for you. Preferred.
2. **Canvas** — multiply every coordinate by `displayedSize / originalSize`.

Never assume the displayed image equals the original resolution.

## Suggested UI

- File input / drag-and-drop that calls `predict(file)`.
- Loading indicator while awaiting the response.
- Toggle for "boxes" vs "highlight" view.
- Per-class colors (map class name → color) if showing multiple mobs.
- A list/legend of detections (class + confidence) beside the image.

## Frontend checklist

- [ ] `API_BASE_URL` read from config/env, not hardcoded.
- [ ] Upload sends `multipart/form-data` with field `file`.
- [ ] Coordinates scaled correctly (SVG `viewBox` or explicit canvas scale).
- [ ] Handles `detections: []` (no mob) and `polygon: []` (box only).
- [ ] Handles request errors and shows a loading state.
- [ ] Highlight/spotlight effect implemented from the polygon.

## Mock response for local development

Develop against this before the backend is ready:

```json
{
  "width": 512,
  "height": 288,
  "detections": [
    {
      "class": "creeper",
      "confidence": 0.88,
      "box": [180, 60, 330, 250],
      "polygon": [[200, 70], [310, 80], [320, 240], [190, 235]]
    }
  ]
}
```
