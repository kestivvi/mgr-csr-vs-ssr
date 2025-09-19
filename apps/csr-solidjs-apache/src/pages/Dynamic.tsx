import type { Component } from 'solid-js';
import { useParams } from '@solidjs/router';

const Dynamic: Component = () => {
  const params = useParams<{ name: string }>();

  return <h1>Hello, {params.name}</h1>;
};

export default Dynamic;
