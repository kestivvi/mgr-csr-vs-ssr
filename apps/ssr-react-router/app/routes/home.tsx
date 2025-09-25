import type { Route } from "./+types/home";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Hello World" },
    { name: "description", content: "Hello World page" },
  ];
}

export default function Home() {
  return (
    <div>
      <h1>Hello World</h1>
    </div>
  );
}
