<script setup lang="ts">
import { ref } from 'vue';
import { useRoute } from 'vue-router';

const route = useRoute();
const name = route.params.name;
const count = ref(0);
</script>

<template>
  <main>
    <h1>Hello World</h1>
    <p>Dynamic ID: {{ name }}</p>
    <div>
      <p>Count: {{ count }}</p>
      <button @click="count++">Increment</button>
    </div>
  </main>
</template>
