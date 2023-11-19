https://www.mayerowitz.io/blog/a-journey-into-shaders

A shader is a small program running on your GPU that takes, at the very least, pixel coordinates as input and spits out a color as output.

## Coordinates is All You Need

<mark>Shaders transform pixel coordinates into colors</mark>, encoded in RGBA—each channel ranging from 0 to 1.

```
varying vec2 vUv;

void main() {
  // Normalized pixel coordinates (from 0 to 1)
  vec2 st = vUv;

  // redish in x, greenish in y
  // Try to modify the following line to have a blue gradient
  // from left to right.
  gl_FragColor = vec4(st.x, st.y, 0.0, 1.0); // RGBA
}
```

出来的效果是：

![[Pasted image 20231119115320.png]]

这个语法中值得注意的几个地方：

- Inputs：可以将 shader 的 input 声明为 varying 或者 uniform；varying 变量对每个像素均不同，而 uniform 对每个像素的取值均相同；
- Coordinate Origin：UV space 坐标从左下角为 (0, 0)；
- Builtin types：vec2, vec3, vec4, mat2, mat3, 等等；
- Swizzling：Accessing elements of a vector? Easy, just use the dot notation (vec2(1, 2).x gives you 1).
- Output：没有 return 语句，每个像素的取值取决于 **gl_FragColor** 变量。

## One Step() Beyond

`step(float threshold, float value)` 函数可以取一个距离值，当大于 threshold 则返回 1 否则为 0。

```
uniform float u_slider;
varying vec2 vUv;

void main() {
    vec2 st = vUv;

    // Distance of the current pixel to the center of the canvas
    float d = distance(st, vec2(0.5));

    // Using step to get a sharp circle
    // s = 1 if d > 0.25, 0 otherwise
    float s = step(0.25, d);

    // Mix the two colors based on the slider
    // color = u_slider * s + (1-u_slider) * d
    float brightness = mix(d, s, u_slider);

    gl_FragColor = vec4(vec3(brightness), 1.0);
}
```

![[Pasted image 20231119120225.png]]

## Signed Distance Functions (SDF)

When you think of shapes, it’s natural to imagine them as a series of connected points. But here’s a twist: you can also represent shapes in terms of their distance to other points in space.

## One and One Makes Another One

## I Like to Move it

可以传递一个 uniform 的 `u_time` 参数告诉它时间。