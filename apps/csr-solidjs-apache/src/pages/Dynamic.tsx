import type { Component } from 'solid-js';
import { useParams } from '@solidjs/router';

const Dynamic: Component = () => {
  const params = useParams<{ name: string }>();

  return (
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {params.name}</p>
    </div>
  );
};

export default Dynamic;
