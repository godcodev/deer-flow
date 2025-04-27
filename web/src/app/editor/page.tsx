// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import ReportEditor from "~/components/editor";

const content = `
# How Many Times Faster is the Speed of Light Compared to the Speed of Sound?

## Key Points
- The speed of light in a vacuum is about 300,000 kilometers per second, while its speed in media like water, glass, and diamond is lower.
- The speed of sound varies greatly depending on the medium. In air, it ranges from 331 m/s at 0 °C to 346.1 m/s at 25 °C, and is much higher in water and solids.
- Calculations show the speed of light is 874030.49 times faster than the speed of sound in air.
- It is 202289.11 times faster than the speed of sound in water.
- And 50300.75 times faster than the speed of sound in steel.

## Overview
This research focuses on determining how many times faster the speed of light is compared to the speed of sound. The speed of light and sound are fundamental physical quantities, and understanding their relationship can provide insights into various natural phenomena and technological applications. The speed of light is known to be extremely fast, but the exact multiple compared to the speed of sound, which varies depending on the medium, is what this study aims to uncover.

## Detailed Analysis
### Speed of Light
- **In a Vacuum**: The speed of light in a vacuum is precisely defined as 299,792.458 kilometers per second, often approximated as 300,000 kilometers per second for scientific calculations.
- **In Other Media**: 
    - In water (refractive index of 1.3), light slows down to about 225,000 kilometers per second.
    - In glass (refractive index of 1.5), its speed is approximately 200,000 kilometers per second.
    - In diamond (refractive index of 2.4), the speed of light is reduced to 125,000 kilometers per second.

### Speed of Sound
- **In Air**:
    - At 0 °C and sea - level, the speed of sound in dry air is about 331 m/s.
    - At 20 °C, it is about 343 m/s.
    - At 25 °C, it is about 346.1 m/s.
- **In Water**:
    - In fresh water at 20 °C, sound travels at about 1481 m/s.
    - In salt water under specific conditions (1000 kilopascals, 10 °C and 3% salinity), it travels at about 1500 m/s.
- **In Other Media**:
    - In iron, sound travels at 5120 m/s.
    - In diamond, sound travels at 12,000 m/s.

### Comparison of Speeds
| Medium | Speed of Light (m/s) | Speed of Sound (m/s) | Multiple (Light/Sound) |
| ---- | ---- | ---- | ---- |
| Air | 299792458 | 343 (at 20 °C) | 874030.49 |
| Water | 299792458 | 1481 (fresh water at 20 °C) | 202289.11 |
| Steel (assumed similar to iron) | 299792458 | 5120 | 50300.75 |


## Key Citations
- [Speed of Light - Evident Scientific](https://evidentscientific.com/en/microscope-resource/knowledge-hub/lightandcolor/speedoflight)
- [Speed of sound - Wikipedia](https://en.wikipedia.org/wiki/Speed_of_sound)
`;

export default function Page() {
  return (
    <main className="flex h-full w-full">
      <div className="flex h-screen flex-auto">
        <ReportEditor content={content} />
      </div>
    </main>
  );
}
