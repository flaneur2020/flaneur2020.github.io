---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
draft: false
description: "Add a brief description here."
---

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vitae justo vel velit sagittis 
tempor. Nullam in dignissim leo, vitae tincidunt nisi. Cras ultrices molestie tortor, 
vel fringilla purus volutpat non.

Proin sit amet purus lectus. Sed dapibus diam vel magna commodo, nec vulputate tellus fermentum. 
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.

## Code

```Python3
if __name__ == "__main__":
    # Check if a file path argument is provided
    if len(sys.argv) < 2:
        print("Usage: ./parse_quic_log.py <path_to_log_file>", file=sys.stderr)
        sys.exit(1)  # Exit with an error code indicating incorrect usage

    input_filepath = sys.argv[1]
    
    output_data = parse_quic_log(input_filepath)

    if output_data["packets"]:
        # Output as a JSON object to standard output
        print(json.dumps(output_data, indent=2))
    elif not output_data["packets"] and os.path.exists(input_filepath) and os.path.getsize(input_filepath) > 0:
        # This case means the file exists and is not empty, but no matching lines were found
        print(f"Warning: No matching data found in '{input_filepath}'.", file=sys.stderr)
        # Still output the empty structure
        print(json.dumps(output_data, indent=2))
```

## Math

You can use LaTeX math in your posts. For inline math, use single dollar signs, e.g. $E = mc^2$.

For display math, use double dollar signs:

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

You can also use environments like align:

$$
\begin{align}
a^2 + b^2 &= c^2 \\
\nabla \cdot \vec{E} &= \frac{\rho}{\varepsilon_0}
\end{align}
$$

Here is a very long LaTeX equation to test horizontal scrolling and container overflow:

$$
\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx = 1 \quad \text{where} \quad \mu \text{ is the mean and } \sigma^2 \text{ is the variance of the normal distribution function, and this equation is intentionally long to test the rendering and scrolling behavior of MathJax in Hugo posts.}
$$

## Picture

You can add images to your post like this:

![Descriptive alt text](/images/example.png)
*This is an image caption that will be centered below the image*
