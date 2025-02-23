import React, { useState, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  useMapEvents,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import { useNavigate } from "react-router-dom";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import pinImage from "./images/pin.png";

const ResizeMap = () => {
  const map = useMap();
  useEffect(() => {
    setTimeout(() => {
      map.invalidateSize();
    }, 100);
  }, [map]);
  return null;
};

// Handles lat and long for interactive map
const MapClickHandler = ({ setLatitude, setLongitude }) => {
  useMapEvents({
    click(e) {
      setLatitude(e.latlng.lat);
      setLongitude(e.latlng.lng);
    },
  });
  return null;
};

const FirePredictionForm = () => {
  const [county, setCounty] = useState("");
  const [month, setMonth] = useState("");
  const [year, setYear] = useState("");
  const [cause, setCause] = useState("");
  const [latitude, setLatitude] = useState(null);
  const [longitude, setLongitude] = useState(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [mapVisible, setMapVisible] = useState(false);
  const [counties, setCounties] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetch("/data/counties.json")
      .then((response) => response.json())
      .then((data) => {
        console.log("Loaded counties:", data);
        setCounties(data);
      })
      .catch((error) => console.error("Error loading counties:", error));
  }, []);

  const causes = [
    "Lightning",
    "Equipment Use",
    "Smoking",
    "Campfire",
    "Debris Burning",
    "Railroad",
    "Arson",
    "Children",
    "Miscellaneous",
  ];
  const months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const payload = {
      county,
      month: months.indexOf(month) + 1,
      year,
      cause,
      latitude,
      longitude,
    };
    try {
      const response = await fetch(
        "https://lemondrop-datathon.onrender.com/predict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      if (response.ok) {
        const data = await response.json();
        navigate("/results", {
          state: { mitigationPlan: data.mitigation_plan },
        });
      } else {
        setMessage("Error: Failed to get prediction");
      }
    } catch (error) {
      setMessage("Error: Could not connect to prediction service");
    } finally {
      setLoading(false);
    }
  };

  const customIcon = new L.Icon({
    iconUrl: pinImage,
    iconSize: [48, 48],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32],
  });

  // Determine if form is valid based on its county or coordinates
  const isFormValid =
    (county || latitude || longitude) && month && year && cause;

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background:
          'url("https://cdn.photoroom.com/v2/image-cache?path=gs://background-7ef44.appspot.com/backgrounds_v3/forest/27_forest.jpg") no-repeat center center fixed',
        backgroundSize: "cover",
        padding: "2rem",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "stretch",
          gap: "1rem",
          width: mapVisible ? "1016px" : "500px",
          transition: "width 0.3s ease-in-out",
        }}
      >
        <div
          style={{
            width: "500px",
            backgroundColor: "white",
            borderRadius: "8px",
            boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
            padding: "2rem",
          }}
        >
          <div style={{ marginBottom: "2rem", textAlign: "center" }}>
            <h1
              style={{
                color: "#1a1a1a",
                fontSize: "24px",
                marginBottom: "8px",
              }}
            >
              Fire Size Prediction
            </h1>
            <p style={{ color: "#666", fontSize: "14px" }}>
              Enter the details below to predict potential fire size
            </p>
          </div>
          <form
            onSubmit={handleSubmit}
            style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}
          >
            <div>
              <label
                style={{
                  display: "block",
                  marginBottom: "8px",
                  color: "#4a4a4a",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
              >
                Location
              </label>
              <select
                value={county}
                onChange={(e) => setCounty(e.target.value)}
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  borderRadius: "4px",
                  border: "1px solid #ddd",
                  fontSize: "14px",
                }}
              >
                <option value="">Select a county or open map</option>
                {counties.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label
                style={{
                  display: "block",
                  marginBottom: "8px",
                  color: "#4a4a4a",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
              >
                Month
              </label>
              <select
                value={month}
                onChange={(e) => setMonth(e.target.value)}
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  borderRadius: "4px",
                  border: "1px solid #ddd",
                  fontSize: "14px",
                }}
              >
                <option value="">Select a month</option>
                {months.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label
                style={{
                  display: "block",
                  marginBottom: "8px",
                  color: "#4a4a4a",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
              >
                Year
              </label>
              <input
                type="number"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                placeholder="Enter year"
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  borderRadius: "4px",
                  border: "1px solid #ddd",
                  fontSize: "14px",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label
                style={{
                  display: "block",
                  marginBottom: "8px",
                  color: "#4a4a4a",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
              >
                Cause
              </label>
              <select
                value={cause}
                onChange={(e) => setCause(e.target.value)}
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  borderRadius: "4px",
                  border: "1px solid #ddd",
                  fontSize: "14px",
                }}
              >
                <option value="">Select cause</option>
                {causes.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            {longitude && latitude && (
              <div
                style={{
                  color: "#4a4a4a",
                  fontSize: "14px",
                  fontWeight: "500",
                }}
              >
                <p>
                  Selected Coordinates: {latitude.toFixed(4)},{" "}
                  {longitude.toFixed(4)}
                </p>
              </div>
            )}
            <button
              type="submit"
              disabled={loading || !isFormValid}
              style={{
                backgroundColor: loading || !isFormValid ? "#ccc" : "#007bff",
                color: "white",
                padding: "12px",
                borderRadius: "4px",
                border: "none",
                fontSize: "14px",
                fontWeight: "500",
                cursor: loading || !isFormValid ? "not-allowed" : "pointer",
                marginTop: "1rem",
              }}
            >
              {loading ? "Processing..." : "Predict Fire Size"}
            </button>
          </form>
          {message && (
            <div
              style={{
                marginTop: "1.5rem",
                padding: "1rem",
                backgroundColor: "#f8f9fa",
                borderRadius: "4px",
                border: "1px solid #dee2e6",
                color: "#4a4a4a",
                fontSize: "14px",
              }}
            >
              {message}
            </div>
          )}
          <button
            type="button"
            onClick={() => setMapVisible(true)}
            style={{
              padding: "8px 12px",
              backgroundColor: "#007bff",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              marginTop: "1rem",
            }}
          >
            Show Map
          </button>
        </div>
        {/* Map Section */}
        {mapVisible && (
          <div style={{ width: "500px" }}>
            <MapContainer
              center={[37.7749, -122.4194]}
              zoom={5}
              style={{ width: "100%", height: "100%", borderRadius: "8px" }}
            >
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              <ResizeMap />
              <MapClickHandler
                setLatitude={setLatitude}
                setLongitude={setLongitude}
              />
              {latitude !== null && longitude !== null && (
                <Marker position={[latitude, longitude]} icon={customIcon}>
                  <Popup>
                    Selected: {latitude.toFixed(4)}, {longitude.toFixed(4)}
                  </Popup>
                </Marker>
              )}
            </MapContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default FirePredictionForm;
