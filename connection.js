// db.js
const { MongoClient } = require("mongodb");

const uri = "mongodb+srv://st4811:12345678UTR@pypark.3exozxa.mongodb.net/";
const client = new MongoClient(uri);

async function conectar() {
  if (!client.topology || client.topology.isClosed()) {
    await client.connect();
    console.log("Conectado a MongoDB");
  }
  return client;
}

async function escucharCambios(callback) {
  try {
    const client = await conectar();
    const db = client.db("parking_monitor");
    const collection = db.collection("estados");

    const changeStream = collection.watch();
    console.log("Escuchando cambios en la colección 'estados'...");

    changeStream.on("change", (change) => {
      if (change.operationType === "insert") {
        const nuevoDoc = change.fullDocument;
        const spots = nuevoDoc.spots;
        callback(spots);
      }
    });
  } catch (error) {
    console.error("Error al escuchar cambios:", error);
  }
}

// Aquí defines qué hacer cuando haya una actualización
escucharCambios((spots) => {
  const spotsStatus = spots.map((spot) => spot.status);

  // Asignar a variables individuales (spot0, spot1, ..., spot9)
  const [spot0, spot1, spot2, spot3, spot4, spot5, spot6, spot7, spot8, spot9] = spotsStatus;

  // Mostrarlos en consola
  console.log("----- NUEVO ESTADO -----");
  console.log("spot0:", spot0);
  console.log("spot1:", spot1);
  console.log("spot2:", spot2);
  console.log("spot3:", spot3);
  console.log("spot4:", spot4);
  console.log("spot5:", spot5);
  console.log("spot6:", spot6);
  console.log("spot7:", spot7);
  console.log("spot8:", spot8);
  console.log("spot9:", spot9);
});
