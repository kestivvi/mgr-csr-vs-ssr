<script setup lang="ts">
import { ref } from 'vue';

const count = ref(0);
</script>

<template>
  <main>
    <h1>Hello World</h1>
    <div>
      <p>Count: {{ count }}</p>
      <button @click="count++">Increment</button>
    </div>
  </main>
</template>
