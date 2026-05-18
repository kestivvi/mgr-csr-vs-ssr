<script setup lang="ts">
defineProps<{ inStock: boolean }>();
</script>

<template>
  <span v-if="inStock" class="badge badge-in">IN</span>
  <span v-else class="badge badge-out">OUT</span>
</template>
