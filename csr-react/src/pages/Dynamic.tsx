import { useParams } from 'react-router-dom';

function Dynamic() {
  const { name } = useParams<{ name: string }>();

  return <h1>Hello, {name}</h1>;
}

export default Dynamic; 