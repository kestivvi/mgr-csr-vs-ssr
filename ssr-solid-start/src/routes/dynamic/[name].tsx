import { useParams } from "@solidjs/router";

export default function HelloWorldPage() {
  const params = useParams<{ name: string }>();
  
  return (
    <div>
      <h1>Hello, {params.name}</h1>
    </div>
  );
}
