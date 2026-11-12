import React from 'react';
import {Composition} from 'remotion';
import {Robot, TOTAL, W, H, FPS} from './Robot';

export const Root: React.FC = () => (
  <Composition id="Robot" component={Robot} durationInFrames={TOTAL} fps={FPS} width={W} height={H} />
);
